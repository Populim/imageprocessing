# Trabalho MVP

# BCC018 7º semestre	bruno gazoni 7585037
# BCC018 7º semestre	matheus steigenberg populim 10734710
# BCC018 7º semestre	Rafael Ceneme 	9898610
# BCC018 7º semestre	Bruno Baldissera  10724351

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#Cálculo do erro quadrático médio
def rmse(A,B):
	N = A.shape[0]
	erro = 0.0

	A = A.astype(np.int32)
	B = B.astype(np.int32)

	for i in range(N):
		for j in range(N):
			erro += (A[i][j] - B[i][j])**2
	return (np.sqrt(erro))/N



# Normaliza uma matriz para o range [0,255] e 
# converte para uint8
def normalize(A):
	a = A.min()
	b = A.max()
	A = ((A-a)/(b-a))*255
	A = A.astype(np.uint8)
	return A


# Função 1: Filtering 1D
def F1(A,F):
	N = A.shape[0] # size of img
	n = F.shape[0] # size of filter
	size = N*N
	B = A.reshape(size)
	B = np.pad(B,n//2,mode='wrap')
	C = np.zeros(size)

	for i in range(size):
		C[i] = np.dot(F,B[i:i+n])

	C = normalize(C)
	C = C.reshape((N,N))
	return C


# Função 2: Filtering 2D
def F2(A,F):
	N = A.shape[0] # size of img
	n = F.shape[0] # size of filter
	B = np.pad(A,n//2,mode='wrap')
	C = np.zeros((N,N),dtype=np.float32)

	for i in range(N):
		for j in range(N):
			C[i,j] = np.sum(np.matmul(F,B[i:i+n,j:j+n]))
			
	C = normalize(C)
	return C


# Função 3: Median Filter
def F3(A,n):
	N = A.shape[0] # size of img
	B = np.pad(A,n//2,mode='wrap')
	C = np.zeros((N,N))

	for i in range(N):
		for j in range(N):
			C[i,j] = np.median(B[i:i+n,j:j+n])
			
	C = normalize(C)
	return C



#Main: lemos parâmetros e abrimos pelo imageio
#os arquivos a serem processados e comparados, em seguida
#chamamos a operação requisitada
def main():
	#print("digite o nome")
	#name1 = input().rstrip()
	name1 = "img10.png"
	A = imageio.imread(name1)

	print(A.shape)

	# print(A[:10,:10,0])
	# print(A[:10,:10,1])
	# print(A[:10,:10,2])

	img_hsv = mpl.colors.rgb_to_hsv(A)


	fig = plt.figure(figsize=(15,10))
	fig.add_subplot(221)
	plt.imshow(A)

	fig.add_subplot(222)
	plt.imshow(img_hsv[:,:,0])

	fig.add_subplot(223)
	plt.imshow(img_hsv[:,:,1])

	fig.add_subplot(224)
	plt.imshow(img_hsv[:,:,2])

	plt.show()


	# F = int(input())
	# n = int(input())
	# if (F == 1):
	# 	filtro = np.zeros(n)
	# 	aux = input().split()
	# 	for i in range(n):
	# 		filtro[i] = float(aux[i])
	# 	C = F1(A,filtro)
	# elif (F == 2):
	# 	filtro = np.zeros((n,n))
	# 	for i in range(n):
	# 		aux = input().split()
	# 		for j in range(n):
	# 			filtro[i,j] = float(aux[j])
	# 	C = F2(A,filtro)		
	# elif(F == 3):
	# 	C = F3(A,n)

	# print(rmse(A,C))


if __name__ == '__main__':
	main()