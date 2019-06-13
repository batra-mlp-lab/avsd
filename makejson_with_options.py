import json
import argparse
import itertools
import numpy as np
import spacy
from tqdm import tqdm
from svqa_utils import multi_sentence_to_vec, sentence_to_vec, get_most_popular, get_data_lists
import random
from multiprocessing import Pool, Queue

random.seed(32)
np.random.seed(32)

def get_options_human_study(answers_train_pop, answers_train_nonpop, gtruth, nearest_answers):
    options = [[] for _ in range(10)]
    options_class = [[] for _ in range(10)]
    for i in range(len(answers_train_pop)):
        random_idx = np.random.randint(10)
        while len(options[random_idx]) >= 10:
            random_idx = np.random.randint(10)
        options[random_idx].append(answers_train_pop[i])
        options_class[random_idx].append('POP')

    if gtruth not in answers_train_pop:
        random_idx = np.random.randint(10)
        while len(options[random_idx]) >= 10:
            random_idx = np.random.randint(10)
        options[random_idx].append(gtruth)
        options_class[random_idx].append('TRUTH')

    option_itr = 0
    for answer in nearest_answers:
        while len(options[option_itr]) >= 10:
            option_itr += 1
            option_itr = option_itr % 10
        if answer not in answers_train_pop:
            options[option_itr].append(answer)
            options_class[option_itr].append('NEAR')
            option_itr+=1
            option_itr=option_itr%10

    ope = options

    nonpop_itr=0
    random.shuffle(answers_train_nonpop)
    for i in range(10):

        while len(options[i]) < 10:
            if (answers_train_nonpop[nonpop_itr] not in nearest_answers) and (answers_train_nonpop[nonpop_itr] != gtruth):
                options[i].append(answers_train_nonpop[nonpop_itr])
                options_class[i].append('RAND')
            nonpop_itr+=1

    options = list(itertools.chain.from_iterable(options))
    options_class = list(itertools.chain.from_iterable(options_class))

    gt_index = options.index(gtruth)
    options_class[gt_index] = 'TRUTH'
    return options, options_class, gt_index

def get_nearest_answer_qspace(question_train_vectors, question_val_vector, answers_train, answer_val, num_nearest_ans):
    distance_vector = np.zeros((question_train_vectors.shape[0],))
    for i in range(question_train_vectors.shape[0]):
        distance_vector[i] = np.linalg.norm(question_val_vector-question_train_vectors[i, :])
    distances_sort = np.argsort(distance_vector)
    nearest_answers = [answer_val]

    count_nearest=0
    for k in range(distances_sort.shape[0]):
        if answers_train[distances_sort[k]] not in nearest_answers:
            nearest_answers = nearest_answers + [answers_train[distances_sort[k]]]
            count_nearest+=1
            if count_nearest == num_nearest_ans:
                break

    return nearest_answers[1:]





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-input_train_json', type=str, default='single_frame/data/train.json')
    parser.add_argument('-input_test_json', type=str, default='single_frame/data/test.json')
    parser.add_argument('-input_val_json', type=str, default='single_frame/data/val.json')
    parser.add_argument('-output_train_json', type=str, default='train_options.json')
    parser.add_argument('-output_test_json', type=str, default='test_options.json')
    parser.add_argument('-output_val_json', type=str, default='val_options.json')
    parser.add_argument('-num_nearest_ans', type=int, default=15)
    parser.add_argument('-num_popular_options', type=int, default=50)
    args = parser.parse_args()

    num_nearest_ans = args.num_nearest_ans
    num_popular_options = args.num_popular_options

    data_train = json.load(open(args.input_train_json, 'r'))
    data_test = json.load(open(args.input_test_json, 'r'))
    data_val = json.load(open(args.input_val_json, 'r'))

    data = []
    counter = 0
    for split in ['train', 'test', 'val']:
        if split == 'train':
            data_split = data_train
        elif split == 'test':
            data_split = data_test
        else:
            data_split = data_val
        for key, value in data_split.items():
            d = {'image_id':key,
                 'dialog':value['data'],
                 'caption':value['script'],
                 'split':split}
            data.append(d)

    answers, questions, images = get_data_lists(data)

    print("number of Answers ", len(answers))
    print("number of Questions ", len(questions))

    answers_pop, answers_nonpop = get_most_popular(answers, num_popular_options)
    
    unique_answers = list(set(answers))
    unique_questions = list(set(questions))

    nlp = spacy.load('en_vectors_web_lg')
    mode = 0
    question_vectors = multi_sentence_to_vec(questions, mode, nlp)
    print('Question_vectors: ', question_vectors.shape)

    def worker(obj, meta):
        pre_a = obj['answer']
        pre_q = obj['question']
        question_vector = sentence_to_vec(pre_q, mode, nlp)
        ans_idx = []
        indices = np.random.choice(question_vectors.shape[0], 4000, replace=False)
        question_vectors_sub = question_vectors[indices]
        answers_sub = [answers[x] for x in indices]
        nearest_answers = get_nearest_answer_qspace(question_vectors_sub, question_vector, answers_sub, pre_a, num_nearest_ans)
        op, op_c, gt_idx = get_options_human_study(answers_pop, answers_nonpop, pre_a, nearest_answers)
        for i in range(100):
            ans_idx = ans_idx + [unique_answers.index(op[i])]
        return (op, op_c, gt_idx, ans_idx), meta

    def unpack_args(args):
        return worker(*args)

    print('Generate options ...')
    obj_list = []
    meta_list = []
    count = 0
    for x in tqdm(range(len(data))):
        dialog = data[x]['dialog']
        split_id = data[x]['split']
        for j in range(10):
            obj_list.append({'question': dialog[j]['question'], 'answer': dialog[j]['answer'], 'id': str(j+1)})
            meta_list.append({'img_id': data[x]['image_id'], 'idx': j, 'split': split_id})
        count += 1
    pool = Pool(64)
    ops_and_meta = list(tqdm(pool.imap(unpack_args, zip(obj_list, meta_list)), total=len(obj_list)))
    ops, meta = zip(*ops_and_meta)

    print('Unpack data and build datasets ...')
    data2_train = {}
    data2_test = {}
    data2_val = {}
    for i in range(len(data)):
        if data[i]['split'] == 'train':
            data2_train[data[i]['image_id']] = data[i]
        elif data[i]['split'] == 'test':
            data2_test[data[i]['image_id']] = data[i]
        else:
            data2_val[data[i]['image_id']] = data[i]

    for i in tqdm(range(len(meta))):
        if meta[i]['split'] == 'train':
            data2_train[meta[i]['img_id']]['dialog'][meta[i]['idx']]['gt_index'] = ops[i][2]
            data2_train[meta[i]['img_id']]['dialog'][meta[i]['idx']]['answer_options'] = ops[i][3]
            
            answer = data2_train[meta[i]['img_id']]['dialog'][meta[i]['idx']]['answer']
            data2_train[meta[i]['img_id']]['dialog'][meta[i]['idx']]['answer'] = unique_answers.index(answer)
            
            question = data2_train[meta[i]['img_id']]['dialog'][meta[i]['idx']]['question']
            data2_train[meta[i]['img_id']]['dialog'][meta[i]['idx']]['question'] = unique_questions.index(question)
        elif meta[i]['split'] == 'test':
            data2_test[meta[i]['img_id']]['dialog'][meta[i]['idx']]['gt_index'] = ops[i][2]
            data2_test[meta[i]['img_id']]['dialog'][meta[i]['idx']]['answer_options'] = ops[i][3]
            
            answer = data2_test[meta[i]['img_id']]['dialog'][meta[i]['idx']]['answer']
            data2_test[meta[i]['img_id']]['dialog'][meta[i]['idx']]['answer'] = unique_answers.index(answer)
            
            question = data2_test[meta[i]['img_id']]['dialog'][meta[i]['idx']]['question']
            data2_test[meta[i]['img_id']]['dialog'][meta[i]['idx']]['question'] = unique_questions.index(question)
        else:
            data2_val[meta[i]['img_id']]['dialog'][meta[i]['idx']]['gt_index'] = ops[i][2]
            data2_val[meta[i]['img_id']]['dialog'][meta[i]['idx']]['answer_options'] = ops[i][3]
            
            answer = data2_val[meta[i]['img_id']]['dialog'][meta[i]['idx']]['answer']
            data2_val[meta[i]['img_id']]['dialog'][meta[i]['idx']]['answer'] = unique_answers.index(answer)
            
            question = data2_val[meta[i]['img_id']]['dialog'][meta[i]['idx']]['question']
            data2_val[meta[i]['img_id']]['dialog'][meta[i]['idx']]['question'] = unique_questions.index(question)
    
    final_data = {'dialogs':data2_train, 'answers':unique_answers, 'questions':unique_questions}
    final_output = {'version': '1.0', 'data': final_data, 'split':'train'}
    print('Writing train data ...')
    with open(args.output_train_json, 'w') as outfile:
        json.dump(final_output, outfile)       

    final_data = {'dialogs':data2_test, 'answers':unique_answers, 'questions':unique_questions}
    final_output = {'version': '1.0', 'data': final_data, 'split':'test'}
    print('Writing test data ...')
    with open(args.output_test_json, 'w') as outfile:
        json.dump(final_output, outfile)
    
    final_data = {'dialogs':data2_val, 'answers':unique_answers, 'questions':unique_questions}
    final_output = {'version': '1.0', 'data': final_data, 'split':'val'}
    print('Writing val data ...')
    with open(args.output_val_json, 'w') as outfile:
        json.dump(final_output, outfile)


