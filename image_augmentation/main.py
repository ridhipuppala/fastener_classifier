
import math
import numpy as np
import numpy.matlib 

X = np.load('dataset.npy')
class_id = np.load('class_id.npy')

class_id = np.array(class_id)

layers =1
class makelayer:
      def __init__(self, name,size):
            self.name = name
            self.size = size

                   
class NN:

    def loaddata(self,path):
        df1 = pd.read_csv(path,header=0)
        M = df1.as_matrix()
        return M
    
    
    def makeNN(self,layers): #call with layers =2
        self.layers =layers
        instancenames = []
        #size =30
        for i in range(0,layers+2):
            instancenames.append(i)
        layer = {name: makelayer(name=name,size=100) for name in instancenames}    
        e = []
        f = []
        layer[0].size = 784
        layer[layers+1].size = 5 
        for i in range(1,layers+2):
            a =layer[i].size 
            c= layer[i-1].size
            e.append(0.01*np.random.randn(c,a))  #weight matrix of dim(784,100)
            f.append(np.zeros((a,1))) #bias
        return e,f
    
    def sigmoid(self,Q):
        M = []
        for i in Q:
            try:
                res = 1.0 /(1.0 + math.exp(-i))
            except OverflowError:
                res = 0.0
            M.append(res)
        return np.transpose(np.asmatrix(M))
    
    def softmax(self,U):
        return np.exp(U)/float(np.sum(np.exp(U)))
    
 
    def feedforward(self,K,W,B):
        #print(K)
        instancenames = []
        for i in range(0,layers+2):
            instancenames.append(i)
        layer = {name: makelayer(name=name,size=100) for name in instancenames}   
        H = []
        A = []
        layer[0].size = 784
        layer[layers+1].size = 5
        for i in range(1,layers+2):
            a =layer[i].size 
            c= layer[i-1].size
            H.append(np.zeros((c,1)))
            A.append(np.zeros((a,1)))
        H.append(np.zeros((5,1)))
        H[0] = np.transpose(np.asmatrix(K))
        for i in range(0,layers):
            A[i] = B[i] + np.matmul(np.transpose(W[i]),H[i])
            H[i+1] = self.sigmoid((A[i]))
        A[layers] = B[layers] + np.matmul(np.transpose(W[layers]),H[layers])
        Y1 = self.softmax((A[layers])) 
        return Y1,H,A
            
    def feature_normalize(self,R):
        mean = np.mean(R)
        range_val = np.amax(R)-np.amin(R)
        R = (R-mean)/float(np.sqrt(np.var(R)))
        return R
    
    def preprocessing(self,Y):
        #convert to one hot vector for prediction
        Y2 = (np.arange(0,5)==Y).astype(float)      
        return np.transpose(np.asmatrix(Y2))    
        
    def sig(self,z):
        try:
            res = 1 / float(1 + math.exp(-z))
        except OverflowError:
            res = 0.0
        return res

    def grad_sig(self,S):
        L = []
        for i in S:
            L.append(self.sig(i)*(1 - self.sig(i)))
        return np.transpose(np.asmatrix(L))    
    
    def back_prop(self,Y3,Y4,H,A,W,B):
        da = [0]*(layers+1)
        db = [0]*(layers+1)
        dw = [0]*(layers+1)
        dh = [0]*(layers+1)
        da[layers] = (Y3-Y4)
        for i in range(layers,-1,-1):
            dw[i] = np.transpose(np.matmul(da[i],np.transpose(H[i])))
            db[i] = da[i]
            dh[i] = np.matmul(W[i],da[i])
            if(i!=0): 
                da[i-1] = np.multiply(dh[i],self.grad_sig(A[i-1]))
        return dw,db
    
    def accuracy(self,U,T):
        acc = np.sum(U == T)/float(len(U))
        return acc
        
    def optimization(self,eta,epochs,batch_size):

        X = np.load('dataset.npy')
        class_id = np.load('class_id.npy')
        Y = np.array(class_id)

        W,B = lays.makeNN(layers)
        for i in range(epochs):
            batches = int((1500/(batch_size))+1)
            for o in range(batches-1):
                db = [0]*(layers+1)
                dw = [0]*(layers+1)
                for j in range(o*batch_size,(o+1)*batch_size):
                    Y2 = self.preprocessing(Y[j])
                    Y1,I,J  = self.feedforward(X[j,:],W,B)
                    print(np.argmax(Y1))
                    dx,dy= self.back_prop(Y1,Y2,I,J,W,B)
                    dw = [sum(x) for x in zip(dw,dx)]
                    db = [sum(x) for x in zip(db,dy)] 
                    LOSS = np.sum((Y1-Y2).^2)/2*15
                for k in range(0,layers+1):
                    W[k] = W[k] - eta*dw[k]
                    B[k] = B[k] - eta*db[k]           
            db = [0]*(layers+1)
            dw = [0]*(layers+1) 
            for m in range(batches*batch_size,len(X)+1):
                Y2 = self.preprocessing(Y[m])
                Y1,I,J   = self.feedforward(X[j,:],W,B)
                dl,dk= self.back_prop(Y1,Y2,I,J,W,B)
                print(np.argmax(Y1))
                dw = [sum(x) for x in zip(dw,dl)]
                db = [sum(x) for x in zip(db,dk)]
            for k in range(0,layers+1):
                W[k] = W[k] - eta*dw[k]
                B[k] = B[k] - eta*db[k]
            print("epoch {}".format(i+1))
        return W,B    



lays = NN()
layers = 2

W,B = lays.optimization(eta=0.01,epochs=50,batch_size=1500)
np.save('weights.npy',W,allow_pickle = True,fix_imports = True)
np.save('bias.npy',B,allow_pickle = True,fix_imports = True)


for i in range(0,784):
     N[:,i] = lays.feature_normalize(N[:,i])


pred = []
for i in range(len(N)):
     P,Q,R = lays.feedforward(N[i,:],W,B)
     pred.append(np.argmax(P))
accur = lays.accuracy(M,np.asmatrix(pred))    
print(accur)



