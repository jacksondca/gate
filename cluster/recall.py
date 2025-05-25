import numpy as np

def read_ivecs(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    ret = a.reshape(-1, d + 1)[:, 1:].copy()
    return ret
 

def read_fvecs(fname):
    return read_ivecs(fname).view('float32')


def read_gt(fname):
    gt = []
    with open(fname) as fp:
        for line in fp.readlines():
            gt.append(list(map(int, line.split())))
    return gt

def calculate_recall(eval_data, gt_data):
    count, total = 0, 0
    
    for eval, gt in zip(eval_data, gt_data):
        count += len(set(eval.tolist()).intersection(set(gt)))
        total += len(gt)
        
    return count / total
    

def recall(ivecs_path, gt_path):
    evaluate_data = read_ivecs(ivecs_path)
    gt = read_gt(gt_path)

    ax, ay = evaluate_data.shape[0], evaluate_data.shape[1]
    bx, by = len(gt), len(gt[0])

    assert (ax == bx) and (ay == by), f'result({ax} x {ay}) != gt({bx} x {by})'
    recall = calculate_recall(evaluate_data, gt)
    return recall


if __name__ == '__main__':
    database = ''
    ivecs_path = f''
    gt_path = f''

    rc = recall(ivecs_path, gt_path)
    print(f'{rc}')
