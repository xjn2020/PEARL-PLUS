import argparse
import json
import os
import re
import math
import operator
import itertools
import torch
import string
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import defaultdict, Counter
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer

MODELS = {
    'bbc': (BertModel, BertTokenizer, 'bert-base-cased'),
    'bbu': (BertModel, BertTokenizer, 'bert-base-uncased'),
    'roberta': (RobertaModel, RobertaTokenizer, 'roberta-base')
}


def check(text):
    if text == '':
        return False

    unwant_list = ['- - -', '&gt', '-.-', '| | |', '-* -*', '^^^', '^^',
                   '-/ -/', '&amp', '\\-\-\\', '......', '-)', 'by by by']
    for char in unwant_list:
        if char in text:
            return False

    for i in range(11):
        if '-' + str(i) in text:
            return False

    for i in range(ord('A'), ord('Z') + 1):
        if '-' + chr(i) in text:
            return False

    for i in range(ord('a'), ord('z') + 1):
        if '-' + chr(i) in text:
            return False

    if len(list(text.split(' '))) < 45:
        return False

    return True


def clean_str(text):
    text = text.replace('.', '')  # old, new

    # clean html1
    clean_links, left_mark, right_mark = [], '&lt;', '&gt;'
    # for every line find matching left_mark and nearest right_mark
    while True:
        next_left_start = text.find(left_mark)
        if next_left_start == -1:
            break
        next_right_start = text.find(right_mark, next_left_start)
        if next_right_start == -1:
            print("Right mark without Left: " + text)
            break
        # print("Removing " + string[next_left_start: next_right_start + len(right_mark)])
        clean_links.append(text[next_left_start: next_right_start + len(right_mark)])
        text = text[:next_left_start] + " " + text[next_right_start + len(right_mark):]
    # print(f"Cleaned {len(clean_links)} html links")

    # clean html2
    pattern = re.compile(r'<[^>]+>', re.S)
    text = pattern.sub(' ', text)

    text = " ".join([s for s in text.split(' ') if "@" not in s])  # clean email, mainly for 20news
    text = re.sub(r"[^A-Za-z0-9(),\(\)=+.!?\"\']", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def has_repeated_words(s, threshold):
    words = s.split()
    counts = Counter(words)
    return any(count for word, count in counts.items() if count / len(words) > threshold)


def load(dataset_dir, dataset_name, lm_type='bbu'):
    docs_from = []
    cleaned_docs, class_names = [], []
    dataset_dir += dataset_name
    with open(os.path.join(dataset_dir, 'dataset.txt'), mode='r',
              encoding='utf-8') as f:
        for raw_doc in f.readlines():
            cleaned_docs.append(clean_str(raw_doc.strip()))  # 5747
    docs_from.append(os.path.join(dataset_dir, 'dataset.txt'))
    print(f"ori docs length: {len(cleaned_docs)}")
    if args.data_form != None:
        if 'ck' in args.data_form:
            with open(os.path.join(dataset_dir, f'df_llama31_gen_{dataset_name}_{args.data_form}.json'), mode='r',
                      encoding='utf-8') as f:
                for i, raw_doc in enumerate(json.load(f)['text']):
                    if check(raw_doc):
                        temp = clean_str(raw_doc.strip())
                        tokens = word_tokenize(temp)
                        comma_count = sum(1 for token in tokens if ',' in token)
                        if len(temp) < 1000:
                            continue
                        if len(tokens) < 10:
                            continue
                        if len(set(tokens)) < len(tokens) / 6:
                            continue
                        if comma_count > len(set(tokens)) / 3:
                            continue
                        cleaned_docs.append(temp)
            docs_from.append(os.path.join(dataset_dir, f'df_llama31_gen_{dataset_name}_{args.data_form}.json'))
        else:
            with open(os.path.join(dataset_dir, f'df_keywords_gen_{dataset_name}_{args.data_form}.json'), mode='r',
                      encoding='utf-8') as f:
                for i, raw_doc in enumerate(json.load(f)['text']):
                    if check(raw_doc):
                        temp = clean_str(raw_doc.strip())
                        tokens = word_tokenize(temp)
                        comma_count = sum(1 for token in tokens if ',' in token)
                        if len(temp) < 1000:
                            continue
                        if len(set(tokens)) < len(tokens) / 6:
                            continue
                        if comma_count > len(set(tokens)) / 3:
                            continue
                        cleaned_docs.append(temp)
            docs_from.append(os.path.join(dataset_dir, f'df_keywords_gen_{dataset_name}_{args.data_form}.json'))
    with open(os.path.join(dataset_dir, 'classes.txt'), mode='r', encoding='utf-8') as f:
        class_names = "".join(f.readlines()).strip().split("\n")

    for doc_from in docs_from:
        print(f'docs_from: {doc_from}')
    if lm_type == 'bbu':  # bert-base-uncased
        cleaned_docs = [x.lower() for x in cleaned_docs]
        class_names = [x.lower() for x in class_names]

    return cleaned_docs, class_names


def prepare_doc(tokenizer, text):
    # define some parameters
    max_tokens = tokenizer.model_max_length - 2  # BERT max input length 512 - 2
    sliding_window_size = max_tokens // 2  # keep semantic
    tokenized_to_id_indicies, tokenids_chunks, tokenids_chunk = [], [], []

    # first basic , second wordpiece
    tokenized_text = tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens)
    # print(tokenized_text)
    for index, token in enumerate(tokenized_text + [None]):  # + <-> extend
        tokens = None
        if token is not None:
            # wordpiece_tokenizer after basic_tokenizer
            tokens = tokenizer.wordpiece_tokenizer.tokenize(token)  # doing -> do + ##ing
            # print(tokens)
        if token is None or len(tokenids_chunk) + len(tokens) > max_tokens:
            tokenids_chunks.append([tokenizer.vocab['[CLS]']] + tokenids_chunk + [tokenizer.vocab['[SEP]']])
            # new chunk
            if sliding_window_size > 0:
                tokenids_chunk = tokenids_chunk[-sliding_window_size:]  # last sliding_window_size tokenids
            else:
                tokenids_chunk = []
        if token is not None:
            # on basic token is split into many wordpiece tokens
            # (chunks idx, start idx in chunk, end idx in chunk)
            tokenized_to_id_indicies.append((len(tokenids_chunks), len(tokenids_chunk),
                                             len(tokenids_chunk) + len(tokens)))
            # corresponding wordpiece tokens
            tokenids_chunk.extend(tokenizer.convert_tokens_to_ids(tokens))

    return tokenized_text, tokenized_to_id_indicies, tokenids_chunks


def tf_idf(docs):
    '''
    Args:
        docs: [[..],[..],[..],..], each element is a word
    Returns:
        list, sort by tf_idf score
    '''
    doc_frequency, word_doc = defaultdict(int), defaultdict(int)
    word_tf, word_idf, word_tf_idf, doc_num = {}, {}, {}, len(docs)

    # occurrence times for each word
    for doc in docs:
        for word in doc:
            doc_frequency[word] += 1

    # how many doc occurrence for each word
    for doc in docs:
        for word in set(doc):
            word_doc[word] += 1

    # tf-idf for each word
    for word in tqdm(doc_frequency, desc='tf-idf'):
        word_tf[word] = doc_frequency[word] / sum(doc_frequency.values())
        word_idf[word] = math.log(doc_num / (word_doc[word] + 1))
        word_tf_idf[word] = word_tf[word] * word_idf[word]

    sorted_words = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)  # list
    return [item[0] for item in sorted_words]


def preprocess_docs(tokenizer, docs, vocab_min_occur):
    '''
    Args:
        tokenizer: BertTokenizer
        docs: list, each element is a cleaned doc in str
        vocab_min_occur: min occur times of each word
    Returns:
        tokenization_info: list, each element is (tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
            tokenized_text: list, text after basic tokenizer
            tokenized_to_id_indicies: list, (chunks idx, start idx in chunk, end idx in chunk)
            tokenids_chunks: list , [101, wordpiece tokens, 102]
        occur_words: set, store words which occur > vocab_min_occur
        tf_idf_words: set, store words which tf-idf is large
    Note that nltk.corpus.stopwords should be download beforehand
    '''

    tokenization_info, counts, clean_docs = [], Counter(), []
    # nltk.download('stopwords')
    for doc in tqdm(docs, desc="preprocess_docs"):
        tokenized_text, tokenized_to_id_indicies, tokenids_chunks = prepare_doc(tokenizer, doc)
        tokenization_info.append((tokenized_text, tokenized_to_id_indicies, tokenids_chunks))
        counts.update(word.translate(str.maketrans('', '', string.punctuation)) for word in tokenized_text)
        clean_docs.append([w for w in tokenized_text if not w in stopwords.words('english')])

    del counts['']
    occur_words = set([k for k, c in counts.items() if c >= vocab_min_occur])
    tf_idf_words = set(tf_idf(clean_docs)[0:len(occur_words)])
    return tokenization_info, occur_words, tf_idf_words


def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def sentence_encode(tokens_id, model, layer):
    '''
    Usage: encode one document use PLM BERT in wordpiece
    Args:
        tokens_id: [101, wordpiece tokens, 102]
        model: BertModel
        layer: 12
    Returns:
        layer_embedding: context wordpiece embedding of one chunk, np.array, (wordpiece num, 768)
    '''
    input_ids = torch.tensor([tokens_id], device=model.device)  # torch.Size([1, 354])
    with torch.no_grad():
        hidden_states = model(input_ids)
    all_layer_outputs = hidden_states[2]  # tuple, 12 elements, each elements [1, 354, 768]
    # squeeze(0) [1, 354, 768] -> [354, 768]
    # [1:-1] -> delete [CLS]/[SEP]
    layer_embedding = tensor_to_numpy(all_layer_outputs[layer].squeeze(0))[1: -1]
    # layer_embedding = tensor_to_numpy(hidden_states[0].squeeze(0))[1: -1]
    # print(hidden_states[0] - all_layer_outputs[layer]) # 0
    return layer_embedding


def handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks):
    '''
    Args:
        model: BertModel
        layer: 12
        tokenized_text: list, text after basic tokenizer
        tokenized_to_id_indicies: list, (chunks idx, start idx in chunk, end idx in chunk)
        tokenids_chunks: [[101, wordpiece tokens, 102],...,[101, wordpiece tokens, 102]]
    Returns:
        context_word_emb of one doc, np.array, (words num, 768)
    '''
    layer_embeddings, word_embeddings = [
        sentence_encode(tokenids_chunk, model, layer) for tokenids_chunk in tokenids_chunks
    ], []
    # average one word's wordpiece embedding to get word embedding
    for chunk_index, start_index, end_index in tokenized_to_id_indicies:
        word_embeddings.append(np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0))
    assert len(word_embeddings) == len(tokenized_text)
    # print(np.array(word_embeddings).shape)  # (324, 768)
    return np.array(word_embeddings)


def process(model, layer, tokenization_info, occur_words, tf_idf_words, class_names):
    '''
    Args:
        model: BertModel
        layer: 12
        tokenization_info: list, each element is (tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
        occur_words: set, store words which occur >= vocab_min_occur
        tf_idf_words: set, store words which tf-idf is large
        class_names: list , each element is an attribute value in str
    Returns:
        id2word: list, vocabulary, id to word
        docs_with_id: [[id,...,id],[...]...] all docs in id format
        docs_with_emb: [nparray,...,nparray] each nparray is a doc's embedding
        static_word_emb: np.array, (word num, 768)
        static_class_emb: np.array (class num, 768)
        class_with_id: [[id,id],[id],...] class name in id format
    '''

    class_words = set(' '.join(class_names).split())  # all words in class_names
    word2id, static_word_emb, word_count = {}, defaultdict(int), defaultdict(int)
    docs_with_id, docs_with_emb = [], []

    # for each document
    for tokenized_text, tokenized_to_id_indicies, tokenids_chunks in tqdm(tokenization_info, desc='process'):
        context_word_emb = handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
        doc_with_id, doc_with_emb = [], []
        for idx, word in enumerate(tokenized_text):
            if word in word2id.keys():
                static_word_emb[word2id[word]] += context_word_emb[idx]
                word_count[word2id[word]] += 1
                doc_with_id.append(word2id[word])
                doc_with_emb.append(context_word_emb[idx])
            else:
                if (word in occur_words and word in tf_idf_words) or word in class_words:
                    word2id[word] = len(word2id)
                    static_word_emb[word2id[word]] += context_word_emb[idx]
                    word_count[word2id[word]] += 1
                    doc_with_id.append(word2id[word])
                    doc_with_emb.append(context_word_emb[idx])
        docs_with_id.append(doc_with_id)
        docs_with_emb.append(np.array(doc_with_emb))

    # get static_word_emb
    for id in static_word_emb.keys():
        static_word_emb[id] = static_word_emb[id] / word_count[id]
    static_word_emb = np.array([emb for id, emb in sorted(static_word_emb.items(), key=lambda d: d[0])])

    # get static_class_emb
    static_class_emb = np.zeros((len(class_names), static_word_emb.shape[1]))
    for i, label in enumerate(class_names):
        words = label.split()
        for word in words:
            # print(word, word2id[word], word_count[word2id[word]])
            static_class_emb[i] += static_word_emb[word2id[word]]
        static_class_emb[i] /= len(words)

    # get class_with_id
    class_with_id = [list(map(lambda x: word2id[x], class_name.split())) for class_name in class_names]

    # get id2word
    id2word = [word for word, id in sorted(word2id.items(), key=lambda d: d[1])]
    return id2word, docs_with_id, docs_with_emb, static_word_emb, static_class_emb, class_with_id


def normalize(weights):
    min_max = np.max(weights) - np.min(weights)
    if min_max == 0:
        min_max = 1e-6
    return (weights - np.min(weights)) / min_max


def soft_max(f):
    return np.exp(f) / np.sum(np.exp(f))


def word_class_sim(class_emb, word_emb):
    '''
    Args:
        class_emb: np.array (class num, 768)
        word_emb: np.array, (word num, 768)
    Returns:
        cos_sim: (class num, word num)
        ranked_wid_to_class: (class num, word num)
    '''

    # CPU will GG
    # tri_dim_class_emb = np.stack([class_emb] * word_emb.shape[0], axis=1)    # (class num, word num, 768)
    # numerator = np.sum(tri_dim_class_emb * word_emb, axis=2)
    # denominator = np.linalg.norm(tri_dim_class_emb, axis=2) * np.linalg.norm(word_emb, axis=1)
    # cos_sim = numerator / denominator
    # ranked_wid_to_class = np.argsort(-cos_sim, axis=1)

    cos_sim = np.zeros((class_emb.shape[0], word_emb.shape[0]))
    for i, class_emb_row in enumerate(class_emb):
        numerator = np.sum(class_emb_row * word_emb, axis=1)
        denominator = np.linalg.norm(class_emb_row) * np.linalg.norm(word_emb, axis=1)
        cos_sim[i] = numerator / denominator

    ranked_wid_to_class = np.argsort(-cos_sim, axis=1)
    return cos_sim, ranked_wid_to_class


def get_class_emb(static_word_emb, static_class_emb, class_with_id, len_word_list, eta, class_names, id2word):
    '''
    Args:
        static_word_emb: np.array, (word num, 768)
        static_class_emb: np.array (class num, 768)
        class_with_id: [[id,id],[id],...] class name in id format
        len_word_list: (10, 40)
        eta: 0.75
    Returns:
        class_emb: np.array (class num, 768)
    '''

    used_wids, finished_class = sum(class_with_id, []), defaultdict(bool)  # default false
    class_emb, class_num = np.zeros_like(static_class_emb), static_class_emb.shape[0]
    cos_sim, ranked_wid_to_class = word_class_sim(static_class_emb, static_word_emb)
    min_len, max_len = len_word_list

    for itera in tqdm(range(max_len), desc="get_class_emb"):
        # add a word for each class's word list
        Tag = True
        for i in range(class_num):
            if finished_class[i]:   continue
            Tag = False
            for id in ranked_wid_to_class[i]:
                if id not in used_wids:
                    class_with_id[i].append(id)
                    # update class_emb
                    class_emb[i] = np.average(static_word_emb[class_with_id[i], :],
                                              weights=normalize(cos_sim[i, class_with_id[i]]), axis=0)
                    used_wids.append(id)
                    break

        if Tag: break

        # update finished class
        if itera >= min_len:
            cur_cos_sim, cur_ranked_wid_to_class = word_class_sim(class_emb, static_word_emb)
            for i in range(class_num):
                if finished_class[i]:   continue
                gjq = len(class_with_id[i])
                if (gjq * 2 - len(set(class_with_id[i] + list(cur_ranked_wid_to_class[i][:gjq])))) / gjq < eta:
                    finished_class[i] = True

    return class_emb


def choose_k_words(docs_with_id, docs_with_emb, class_emb, key_num, fd, id2word):
    '''
    Args:
        docs_with_id: [[id,...,id],[...]...] all docs in id format
        docs_with_emb: [nparray,...,nparray] each nparray is a doc's context embedding
        class_emb: np.array (class num, 768)
        key_num: num-keywords 60
        freedom_degree: 1
    Returns:
        new_docs_with_id: [[id,...,id],[...]...] all docs in id format, len(doc) <= num-keywords
        new_docs_with_emb: [nparray,...,nparray] each nparray is a doc's context embedding
        weights: [nparray,...] corresponding normalized weight
    '''

    # f = open("./doc_words.txt","w")
    new_docs_with_id, new_docs_with_emb, weights = [], [], []

    # for each document
    for i, doc_with_emb in enumerate(tqdm(docs_with_emb, desc="choose K words")):
        # compute \pi_{i,j} in formula 8
        tri_dim_word_emb = np.stack([doc_with_emb] * class_emb.shape[0], axis=1)  # (word_num,class num,768)
        gjq = 1 / np.power(1 + np.sum(np.square(tri_dim_word_emb - class_emb), axis=2) / fd, (fd + 1) / 2)
        doc_weights = np.max(gjq, axis=1) / np.sum(gjq, axis=1)

        # choose k words for current document
        idxs = (np.argsort(-doc_weights)[:key_num])
        new_docs_with_id.append([docs_with_id[i][idx] for idx in idxs])
        # f.write(" ".join([id2word[docs_with_id[i][idx]] for idx in idxs]) + '\n')
        new_docs_with_emb.append(doc_with_emb[idxs, :])
        weights.append(normalize(np.array([doc_weights[idx] for idx in idxs])))

    return new_docs_with_id, new_docs_with_emb, weights


def get_save_new_vocab(docs_with_id, id2word, dataset_dir):
    '''
    Args:
        docs_with_id: [[id,...,id],[...]...] all docs in id format, len(doc) <= num-keywords
        id2word: list, vocabulary, id to word
        dataset_dir: ../../data/datasets/profession
    Returns:
        new_id2word: list, vocabulary, id to word
        new_docs_with_id: [[id,...,id],[...]...] all docs in id format, len(doc) <= num-keywords, id from 0
    '''

    new_word2id, new_docs_with_id = {}, []
    for doc in tqdm(docs_with_id, desc="get new vocab"):
        new_doc_with_id = []
        for id in doc:
            word = id2word[id]
            if word not in new_word2id:
                new_word2id[word] = len(new_word2id)
            new_doc_with_id.append(new_word2id[word])
        new_docs_with_id.append(new_doc_with_id)

    # save new_docs_with_id and new_id2word
    if args.data_form is None:
        s = f'info_{args.niter}_{args.num_keywords}'
    else:
        s = f'info_{args.data_form}'
    save_dir = os.path.join(dataset_dir, s)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "doc_wids.txt"), "w", encoding="utf-8") as f:
        for doc in new_docs_with_id:
            f.write(' '.join(map(str, doc)) + '\n')

    with open(os.path.join(save_dir, "voca.txt"), "w", encoding="utf-8") as f:
        for word, id in sorted(new_word2id.items(), key=lambda d: d[1]):
            f.write(str(id) + '\t' + word + '\n')


def init_cos_sim_mat(docs_with_emb, weights, class_emb, dataset_dir):
    '''
    Args:
        docs_with_emb: [nparray,...,nparray] each nparray is a doc's context embedding
        weights: [nparray,...] corresponding normalized weight
        dataset_dir: ../../data/datasets/profession
    Returns:
        generate one cos_sim_mat for one document
    '''
    if args.data_form is None:
        s = f'info_{args.niter}_{args.num_keywords}'
    else:
        s = f'info_{args.data_form}'
    save_dir = os.path.join(dataset_dir, s)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'cos_sim.txt'), 'w', encoding='utf-8') as f:
        # iterate all documents
        for doc_id, doc_with_emb in enumerate(tqdm(docs_with_emb, desc="init cos_sim mat")):
            # iterate all biterms in current document
            for id1, id2 in itertools.combinations(list(range(doc_with_emb.shape[0])), 2):
                biterm_emb = np.average(doc_with_emb[[id1, id2], :], weights=weights[doc_id][[id1, id2]], axis=0)
                numerator = np.sum(class_emb * biterm_emb, axis=1)
                denominator = np.linalg.norm(biterm_emb) * np.linalg.norm(class_emb, axis=1)
                biterm_cos = numerator / denominator
                f.write(" ".join(map(str, biterm_cos)) + '\n')


def main(args):
    #### construct tokenizer/PLM
    model_class, tokenizer_class, pretrained_weights = MODELS[args.lm_type]
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    #### read data
    docs, class_names = load(args.datasets_dir, args.dataset_name, args.lm_type)
    # print(len(docs[18223]))
    print(f"{args.dataset_name}: {len(docs)} docs, {len(class_names)} classes")
    # print(tokenizer.vocab['[CLS]'])
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    model.eval()
    if args.use_gpu: model.cuda()  # cuda:0
    # print("PLM: BERT-bbu, tokenizer: BertTokenizer")

    #### process docs and get many things
    tokenization_info, occur_words, tf_idf_words = preprocess_docs(tokenizer, docs, args.vocab_min_occur)
    id2word, docs_with_id, docs_with_emb, static_word_emb, static_class_emb, class_with_id = process(model, args.layer,
                                                                                                     tokenization_info,
                                                                                                     occur_words,
                                                                                                     tf_idf_words,
                                                                                                     class_names)
    # get class_emb (V_{c_q] in paper) / choose K words for each document / get new vocabulary
    class_emb = get_class_emb(static_word_emb, static_class_emb, class_with_id, args.len_word_list, args.eta,
                              class_names, id2word)
    docs_with_id, docs_with_emb, weights = choose_k_words(docs_with_id, docs_with_emb, class_emb, args.num_keywords,
                                                          args.freedom_degree, id2word)
    # must recode vocabulary, because btm p(w|z) use
    get_save_new_vocab(docs_with_id, id2word, args.datasets_dir + args.dataset_name)
    # init and save cos sim matrix
    init_cos_sim_mat(docs_with_emb, weights, class_emb, args.datasets_dir + args.dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=str, default='./data/')
    parser.add_argument("--dataset-name", type=str, default='20News')  # required=True
    parser.add_argument("--lm-type", type=str, default='bbu')
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--vocab-min-occur", type=int, default=2)  # min occur times of each word
    parser.add_argument("--eta", type=float, default=0.75)  # threshold in word list
    parser.add_argument("--len-word-list", type=tuple, default=(11, 41))  # word list length
    parser.add_argument("--num-keywords", type=int, default=60)  # how many keywords in one doc
    parser.add_argument("--freedom-degree", type=int, default=1)  # lambda in paper
    parser.add_argument("--use-gpu", type=bool, default=True)
    parser.add_argument("--data-form", type=str, default=None)
    parser.add_argument("--screen", action='store_true')
    parser.add_argument("--niter", type=int, default=50)
    # parser.add_argument("--num-BTM", type=int, default=20)  # E in paper
    # parser.add_argument("--num-Gibbs", type=int, default=50)    # T in paper
    args = parser.parse_args()
    print(args)
    # print(vars(args))
    main(args)
