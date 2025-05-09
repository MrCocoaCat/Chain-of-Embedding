from multiprocessing import Pool
import requests
import json
import os
import time

def get_page(id):
    print('<进程%s> get %s' %(os.getpid(),id))
    time.sleep(1)
    #
    return id*id



def pasrse_page(res):
    print('<进程%s> parse %s' %(os.getpid(),res['url']))
    parse_res='url:<%s> size:[%s]\n' %(res['url'],len(res['text']))
    with open('db.txt','a') as f:
        f.write(parse_res)


if __name__ == '__main__':
    task_list = list(range(5))

    p=Pool(3)
    res_l=[]
    for url in task_list:
        res=p.apply_async(get_page,args=(url,))
        res_l.append(res)

    p.close()
    p.join()
    print([res.get() for res in res_l]) #拿到的是get_page的结果,其实完全没必要拿该结果,该结果已经传给回调函数处理了

