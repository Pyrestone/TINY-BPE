import numpy as np
import re
import time

class tiny_bpe:
    def __init__(self,exclude_spaces=True):
        self.exclude_spaces=exclude_spaces
        
    def normalize_text(self,text):
        pattern=re.compile("[^a-zA-Z0-9\' ]")
        z=pattern.sub(" ",text.upper())
        z=re.sub("[ ]+"," ",z)
        return z

    def perform_merge(self,x1,x2,x_merged,text_num,delete=False,merge_prob=1.0):
        new_text=[]
        i=0
        if merge_prob>=1.0:
            while i<(len(text_num)-1):
                if text_num[i]==x1 and text_num[i+1]==x2:
                    new_text.append(x_merged)
                    i+=2
                else:
                    new_text.append(text_num[i])
                    i+=1
        else:
            while i<(len(text_num)-1):
                if text_num[i]==x1 and text_num[i+1]==x2 and np.random.uniform()<merge_prob:
                    new_text.append(x_merged)
                    i+=2
                else:
                    new_text.append(text_num[i])
                    i+=1
        if i==len(text_num)-1:
            new_text.append(text_num[i])
        if delete:
            del text_num
        return np.array(new_text)

    def fit_vocab(self,initial_text, num_merges,verbose=0):
        self.charset=set(list(initial_text))
        self.num_to_tok=dict(list(enumerate(self.charset)))
        self.tok_to_num=dict(map(reversed,self.num_to_tok.items()))
        text_num=np.array([self.tok_to_num[c] for c in initial_text],dtype="int32")
        if verbose>0:
            print(len(text_num))
        self.merges=[]
        self.num_tokens=len(self.charset)
        self.mat=np.zeros((num_merges+self.num_tokens,num_merges+self.num_tokens),dtype="int32")

        def count_tuples():
            self.mat[:self.num_tokens,:self.num_tokens]=0
            for i in range(len(text_num)-1):
                self.mat[text_num[i],text_num[i+1]]+=1

        self.space_char=self.tok_to_num[" "]

        def find_best_merge(mat):
            if self.exclude_spaces:
                mat[:,self.space_char]=0
                mat[self.space_char,:]=0
            best_merge=np.unravel_index(np.argmax(mat),mat.shape)
            merge_count=mat[best_merge[0],best_merge[1]]
            return best_merge,merge_count

        for i in range(num_merges):
            t0=time.time()
            count_tuples()
            t1=time.time()
            best_merge,merge_count=find_best_merge(self.mat)
            if merge_count==0:
                break
            x1,x2=best_merge
            merged_tok=self.num_to_tok[x1]+self.num_to_tok[x2]
            if merged_tok in self.tok_to_num:
                merged_num=self.tok_to_num[merged_tok]
            else:
                merged_num=self.num_tokens
                self.num_to_tok[merged_num]=merged_tok
                self.tok_to_num[merged_tok]=merged_num
                self.num_tokens+=1
            self.merges.append((x1,x2,merged_num))
            t2=time.time()
            text_num=self.perform_merge(x1,x2,merged_num,text_num,delete=True)
            t3=time.time()
            if verbose>0:
                print(i,repr(self.num_to_tok[x1]),"+",repr(self.num_to_tok[x2]),"->",repr(self.num_to_tok[merged_num]),f"({merge_count})")
            if verbose>1:
                print("new length:",len(text_num))
                print(f"count took {t1-t0:.2f}s")
                print(f"best took {t2-t1:.2f}s")
                print(f"merge took {t3-t2:.2f}s")
        if verbose>0:
            print("VOCAB COMPLETED!")

    def print_tok(self,test_text,merge_prob=1.0):
        test_text_num=self.tokenize(test_text,merge_prob)
        for c in test_text_num:
            print(self.num_to_tok[c],end="|")
        print()
    
    def tokenize(self,test_text,merge_prob=1.0):
        test_text_num=np.array([self.tok_to_num[c] for c in test_text.upper()])

        for m1,m2,m3 in self.merges:
            test_text_num=self.perform_merge(m1,m2,m3,test_text_num,merge_prob=merge_prob)
        return test_text_num
    
    def string_tokenize(self,test_text,merge_prob=1.0):
        out=[]
        test_text_num=self.tokenize(test_text,merge_prob)
        for c in test_text_num:
            out.append(self.num_to_tok[c])
        return out

    def serialize(self):
        out={}
        for k in ["num_to_tok","tok_to_num","merges","num_tokens"]:
            if k!="merges":
                out[k]=self.__dict__[k]
            else:
                out["merges"]=[(int(a),int(b),(int(c))) for a,b,c in self.merges]
        return out
    
    def deserialize(self,params):
        self.__dict__=params
        for k in list(self.num_to_tok.keys()):
            self.num_to_tok[int(k)]=self.num_to_tok[k]