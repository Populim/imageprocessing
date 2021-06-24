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



# Função 2: Filtering 2D
def F2(A,F):
	N = A.shape[0] # size of img
	n = F.shape[0] # size of filter
	B = np.pad(A,n//2,mode='wrap')
	C = np.zeros((N,N),dtype=np.float32)

	for i in range(N):
		for j in range(N):
			C[i,j] = np.sum(F * B[i:i+n,j:j+n])
			
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

	teste = 1-img_hsv[:,:,1]


	# filtrox = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	# print(filtrox)

	# filtroy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	# print(filtroy)

	# Cx = F2(teste,filtrox)

	# Cy = F2(teste,filtroy)	


	# fig = plt.figure(figsize=(15,10))

	# fig.add_subplot(221)
	# plt.imshow(teste,cmap='gray')


	# fig.add_subplot(222)
	# plt.imshow(Cx,cmap='gray')

	# fig.add_subplot(223)
	# plt.imshow(Cy,cmap='gray')

	# C = np.sqrt(np.power(Cy,2) + np.power(Cx,2))
	# C = normalize(C)
	# fig.add_subplot(224)
	# plt.imshow(C,cmap='gray')

	# plt.show()


	# print(teste.min(),teste.max())

	f_tr = np.ones(teste.shape).astype(np.uint8)
	# setting to 0 the pixels below the threshold
	f_tr[np.where(teste < 0.94)] = 0

	plt.imshow(f_tr,cmap='gray')
	plt.show()



	# f, (ax1, ax2) = plt.subplots(2)
	# cnts, bins = np.histogram(teste, bins='auto')
	# ax1.bar(bins[:-1] + np.diff(bins) / 2, cnts, np.diff(bins))
	# ax2.hist(teste, bins='auto')

	# hist,_ = np.histogram(teste, bins=20)
	# plt.bar(np.linspace(0.0, 1.0, num=20), hist)

	plt.show()

	plt.imshow(img_hsv[:,:,1],cmap='gray')
	plt.show()



	# fig = plt.figure(figsize=(15,10))
	# fig.add_subplot(221)
	# plt.imshow(A)

	# fig.add_subplot(222)
	# plt.imshow(img_hsv[:,:,0])

	# fig.add_subplot(223)
	# plt.imshow(img_hsv[:,:,1])

	# fig.add_subplot(224)
	# plt.imshow(img_hsv[:,:,2])

	# plt.show()


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