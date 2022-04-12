import numpy as np                           # This library which helps in writing a variety of mathematical notations
import matplotlib.pyplot as plt              # This library helps plot graphs by making two lists/arrays as x and y axis
import csv                                   # This helps in reading csv files

b=[]                                         
a=[]                                         
x=[]                                          
y=[]

path=input('Please provide path of data file : ')

with open(path,'r') as csvfile:
    lines=csv.reader(csvfile, delimiter=',')     # csv stands for comma seperated values, so ',' is used as delimiter
    for row in lines:
        a.append(row[0])
        b.append(row[1])

a.pop(0)
b.pop(0)

for t in a:
    x.append(float(t))      # We get list of all values of x[n] in float format from the csv file, stored in x
    
for m in b:
    y.append(float(m))      # We get list of all values of y[n] in float format from the csv file, stored in y
    
n=[t for t in range(1,len(y)+1)]  # List against which x[n] and y[n] will be plotted

# Plotting Given Data

plt.plot(n,x,label='x[n]')
plt.plot(n,y,label='y[n]')
plt.xlabel('Place Number')
plt.ylabel('Temprature in degree Celsius')
plt.legend()
plt.title('Given Data')
plt.show()

#=======================================================================================================================================================

h=[1/16,4/16,6/16,4/16,1/16]       # Impulse response for blurring

# Defining DTFT, note that we have taken 10000 distinct Ω [0,2π) to increase accuracy, which are 2π(0/10000), 2π(1/10000), 2π(2/10000) ... 2π(9999/10000)
# Now since there are so many values so calculation will take time, please bear with us

def DTFT(func):
    F1=[]
    m=1000
    s = [2*i*np.pi/(m) for i in range(m)]
    for i in s:
        p = 0
        for x in range(len(func)):
            #print(x)
            p += func[x]*np.exp( (-1j*i*x) )
        F1.append(p)
    return F1

n1=[n for n in range(1000)]      # List against which all DTFTs will be plotted 

H=DTFT(h)                         # DTFT of h[n]
Y=DTFT(y)                         # DtFT of y[n]

# Plotting DTFT of h[n]

plt.plot(n1,Y)
plt.title('Y : DTFT of y[n]')
plt.show()

# Plotting DTFT of y[n]

plt.plot(n1,H)
plt.title('H : DTFT of h[n]')
plt.show()

# Defining Inverse of DTFT, here r is number of elements in list of which DTFT is to be inversed

def InvDTFT(y,r):
    n=len(y)
    l=[]
    for r in range(r):
        s=0
        for i in range(n):
            exp=np.exp(2j * np.pi * r * i / n)   #calculating the exponential value of all elements
            s+=y[i]*exp
        l.append(s)
    return ((1 / n) * np.array(l))
  
# Checking the Inverse DTFT of DTFTs (Proves that both DTFT and Inverse of DTFT are accurate)

h_inv=InvDTFT(H,5)      # Here r=5, as number of elements in h[n] is 5 and this h_inv should be same as h[n]
plt.plot([-2,-1,0,1,2],h_inv)  # Mapping such that 6/16 corresponds to n=0
plt.title('Inverse DTFT of H (H is DTFT of h[n])')
plt.show()

y_inv=InvDTFT(Y,193)    # Here r=193, as number of elements in y[n] is 193 and this y_inv should be same as y[n]
plt.plot(n,y_inv)
plt.title('Inverse DTFT of Y (Y is DTFT of y[n])')
plt.show()

#===========================================================================================================================================================


# Defining function for denoising

def denoise(func):
    
    f=[]                     # Creating a new list f so that original list is not altered
    for i in func:
        f.append(i)       
    
    # Getting values of index 1,2 and 190,191 so as to extend list f so that average can be taken for all
    
    rep1=f[1]                
    rep2=f[2]
    rep3=f[len(f)-2]
    rep4=f[len(f)-3]

    f.insert(0,rep1)         # This adds the value of index 1 at 0 index, shifting rest all 
    f.insert(0,rep2)         # This adds the value of index 2 at 0 index, shifting rest all
    f.append(rep3)           # This adds the value of index -2 at end
    f.append(rep4)           # This adds the value of index -3 at end

    i=0
    denoised=[]
    
    while i < len(f)-4:      # This revolves such that all initial 193 values are covered when at centre, all having averege of 5 values
        denoised.append((f[i]+f[i+1]+f[i+2]+f[i+3]+f[i+4])/5)
        i+=1
    return denoised

plt.plot(n,y,label='y[n]')
y1=denoise(y)
plt.plot(n,y1,label='Denoised alone')
plt.title('Denoise only')
plt.legend()
plt.show()

#===========================================================================================================================================================

#Defining deblur function

def deblur(func):
    

    f=[]                     # Creating a new list f so that original list is not altered  
    for i in func:
        f.append(i)

    
    rep1=f[1]                
    rep2=f[2]
    rep3=f[len(f)-2]
    rep4=f[len(f)-3]
    
    f.insert(0,rep1)         # This adds the value of index 1 at 0 index, shifting rest all 
    f.insert(0,rep2)         # This adds the value of index 2 at 0 index, shifting rest all
    f.append(rep3)           # This adds the value of index -1 at end
    f.append(rep4)           # This adds the value of index -2 at end
    

    i=0
    deblurred=[]
    
    while i < len(f)-4:
        
        Func_DTFT=[]
        
        l=[f[i],f[i+1],f[i+2],f[i+3],f[i+4]]
        L=DTFT(l)
        
        for t in range(len(L)):
            if abs(H[t])<0.18:
                Func_DTFT.append(abs(L[t])/0.18)      # Division by less than 0.18 will make huge values so we fix it here, also since its this close so it provides maximum accuracy
            else:
                u=abs(L[t])/abs(H[t])
                Func_DTFT.append(u)
                
        x_deblur=np.array(InvDTFT(Func_DTFT,5))
        deblurred.append(sum(x_deblur)/5)
        i=i+1
    
    return deblurred
        

plt.plot(n,y,label='y[n]')
y2=deblur(y)
plt.plot(n,y2,label='Deblurred alone')
plt.title('Deblur only')
plt.legend()
plt.show()

# ====================================== Part A : First Denoise then Deblur ========================================

x1=deblur(y1)   # y1 was denoise only y[n]

plt.plot(n,x,label='x[n]')
plt.plot(n,x1,label='x1[n]')
plt.title('Part A : First Denoise then Deblur')
plt.legend()
plt.show()



#====================================== Part B : First Deblur then Denoise =========================================

x2=denoise(y2)   # y2 was deblur only y[n]

plt.plot(n,x,label='x[n]')
plt.plot(n,x2,label='x2[n]')
plt.title('Part B : First Deblur then Denoise')
plt.legend()
plt.show()

#=========================================================================================

print('Sum Square Diffrence (SSD) of x1[n] with respect to x[n] : ',abs(np.sum(np.square(np.array(x) - np.array(x1)))))
print('Sum Square Diffrence (SSD) of x2[n] with respect to x[n] : ',abs(np.sum(np.square(np.array(x) - np.array(x2)))))

# Since SSD of x2[n] is less than x1[n] so we can say mathematically first denoising and then deblurring is better

