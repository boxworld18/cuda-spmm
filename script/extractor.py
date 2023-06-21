import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, help='file')
    args = parser.parse_args()

    lines = []
    print('file = ', args.f)
    with open(args.f, 'r') as f:
        lines = f.readlines()
    
    ds_name = []
    test_res = []
    res_cu = []
    res_opt = []
    
    cnt = 0
    failed = False
    is_failed = False
    for line in lines:
        if 'dset' in line:
            ds_name.append(line.split('"')[1])
            test_res.append(failed)
            cnt = 0
            failed = False
        if 'time = ' in line:
            ti = line.split('time = ')[1].split(' ')[0]
            if cnt == 0:
                res_cu.append(float(ti))
            else:
                res_opt.append(float(ti))
            cnt += 1
        if 'FAILED' in line:
            is_failed = True
            failed = True
    
    test_res.append(failed)
    test_res.remove(0)
    
    for i in range(len(ds_name)):
        print('{:<15} {:>15.8f} {:>15.8f} {:>15.6f} {}'.format(ds_name[i], res_cu[i], res_opt[i], res_cu[i] / res_opt[i], '' if test_res[i] == False else 'FAILED'))

    if is_failed:
        print('FAILED')