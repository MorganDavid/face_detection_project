import pickle
# Combines two pkl databases. 
_file1 = r'train_neg.pkl'
_file2 = r'train_pos.pkl'
_out = r'train_db.pkl'
with open(_file1,'rb') as f:
	f1 = pickle.load(f)
with open(_file2,'rb') as f:
	f2 = pickle.load(f)
print("f1",len(f1))
print("\n\n\n f2:",f2)


out = f1+f2
print("length:",len(out))
#print("\n\n\n out:",out)
with open(_out,'wb') as f:
	pickle.dump(out, f)
