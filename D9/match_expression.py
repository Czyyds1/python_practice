import re

def a_match():
    result=re.match('.','abc')
    print(result.group())
    result=re.match('si.s','sigs')
    print(result.group())
    result=re.match('ab.','abcab')
    print(result.group())
    result=re.match('[Xx]', 'XxYz')
    print(result.group())
    result=re.match('[Xx]','xyz')
    print(result.group())
    result=re.match('[0-4]abc','3abc')
    print(result.group())
    result=re.match('[0-35-9]', '369abc')
    print(result.group())
    result=re.match(r'丢了\d块钱','丢了5块钱啦')
    print(result.group())

def multiple_match():
    res=re.match(r'[a-z][0-9]*', 'a1')
    print(res.group())
    res=re.match(r'[a-z]?[0-9]','a33')
    print(res.group())
    res = re.match(r'[a-zA-Z0-9_]{10}', '1a2b3c4d5e6f7g8h9i0j')
    print(res.group())

def group_match():
    res=re.match(r'[1-9]?\d$|100','3')
    print(res.group())
    res=re.match(r'[04]?\d$|100','100')
    print(res.group())
    res = re.match(r'\w{4,20}@(163|gmail|qq)\.com', 'test@gmail.com')
    print(res.group())






if __name__ == '__main__':
    a_match()
    multiple_match()
    group_match()