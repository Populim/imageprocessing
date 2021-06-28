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


def normalize_one(A):
	a = A.min()
	b = A.max()
	A = ((A-a)/(b-a))
	return A


# Normaliza uma matriz para o range [0,255] e 
# converte para uint8
def normalize(A):
	a = A.min()
	b = A.max()
	A = ((A-a)/(b-a))*255
	A = A.astype(np.uint8)
	return A

def gamma_correction(A,gamma):
	B = np.zeros((A.shape))
	B = 255*(np.power(A/255.0,1.0/gamma))
	B = normalize(B)
	return B

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

def get_average_filter(A):
	N = A.shape[0] # size of img
	n = 100
	C = np.zeros((N//n,N//n),dtype=np.float32)


	for i in range(N//n):
		for j in range(N//n):
			C[i,j] = np.average(A[i*n:(i+1)*n,j*n:(j+1)*n])


	C = np.kron(C,np.ones((n,n)))

	return C


def main():
	name1 = "img10.png"
	A = imageio.imread(name1)

	print(A.shape)

	img_hsv = mpl.colors.rgb_to_hsv(A)

	teste = 1-img_hsv[:,:,1]


	media = get_average_filter(teste)

	teste = normalize_one(teste - media)
	print(teste)
	teste_gamma = gamma_correction(teste,0.8)
	print(teste_gamma)


	valor = img_hsv[:,:,2]
	print(valor, "aqui em cima")
	print("aqui")

	f_tr = np.ones(teste.shape).astype(np.uint8)
	# setting to 0 the pixels below the threshold
	# f_tr = A[:,:,0]
	A[(np.where((teste_gamma > 180) & (valor < 200) & (valor > 90)))] = [255,0,0]

	plt.imshow(A)
	plt.show()



	plt.show()




if __name__ == '__main__':
	main()