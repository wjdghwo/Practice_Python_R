#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[16]:


array1 = np.array([1,2,3])
print('array1 type:', type(array1))
print('array1 array 형태:', array1.shape)


# In[17]:


array2 = np.array([[1,2,3],
                 [2,3,4]])
print('array2 type:', type(array2))
print('array2 array 형태:', array2.shape)


# In[18]:


array3 = np.array([[1,2,3]])
print('array3 type:', type(array3))
print('array3 array 형태:', array3.shape)


# In[22]:


print('array1: {:0}차원, array2: {:1}차원, array3: {:2}차원'.format(array1.ndim,
                                                               array2.ndim,array3.ndim))


# In[28]:


#int32 정수, float64 실수
list1=[1,2,3]
print(type(list1))
array1=np.array(list1)
print(type(array1))
print(array1,array1.dtype)


# In[31]:


list2=[1,2,'test']
array2=np.array(list2)
print(array2,array2.dtype)


# In[32]:


list3=[1,2,3.0]
array3=np.array(list3)
print(array3,array3.dtype)


# In[3]:


array_int = np.array([1,2,3])
array_float = array_int.astype('float64')
print(array_float,array_float.dtype)


# In[5]:


array_int1=array_float.astype('int32')
print(array_int1,array_int1.dtype)


# In[9]:


array_float1=np.array([1.1,2.1,3.1])
array_int2=array_float1.astype('int32')
print(array_int2, array_int2.dtype)


# In[10]:


sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)


# In[11]:


zero_array=np.zeros((3,2), dtype='int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)


# In[12]:


one_array=np.ones((3,2))
print(one_array)
print(one_array.dtype, one_array.shape)


# In[24]:


#reshape 벡터를 행렬화 : 차원 및 크기 변경
array1=np.arange(10)
print('array1\n:', array1)

array2=array1.reshape(2,5)
print('array2\n:', array2)

array3=array1.reshape(5,2)
print('array3\n:', array3)

#지정된 사이즈 외에는 변경불가
array1.reshape(4,3)


# In[34]:


#reshape에 -1을 얺으면 호환되는 새로운 shape로 변환
array1 = np.arange(10)
print(array1)

array2 = array1.reshape(-1,5)
print('array2 shape:', array2.shape)

array3 = array1.reshape(5,-1)
print('array3 shape:', array3.shape)

#지정된 사이즈 외에는 변경불가
array4=array1.reshape(1-,4)


# In[35]:


array1=np.arange(8)
array3d=array1.reshape((2,2,2))
print('array3d:\n', array3d.tolist())
#tolist(): 리스트 자료형으로 변환


# In[39]:


#3차원ndarray를 2차원 ndarray로 변환
array5 = array3d.reshape(-1,1)
print('array5:\n', array5.tolist())
print('array5 shape:', array5.shape)


# In[40]:


#1차원ndarray를 2차원 ndarray로 변환
array6 = array1.reshape(-1,1)
print('array6:\n', array6.tolist())
print('array6 shape:', array6.shape)


# In[49]:


#1부터 9까지의 1차원 ndarray 생성
array1 = np.arange(start=1,stop=10)
print('array1:',array1)

#index는 0부터 시작하므로 array1[2]는 3번째 index 위치의 데이터값을 의미
value = array1[2]
print('value:',value)
print(type(value))

print('맨 뒤의 값:', array1[-1], ', 맨 뒤에서 두 번째 값:', array1[-2])


# In[51]:


array1[0]=9
array1[8]=0
print('array1:',array1)


# In[54]:


array1d = np.arange(start=1,stop=10)
array2d = array1d.reshape(3,3)
print(array2d)

print('(row=0, col=0) index 가리키는 값:', array2d[0,0])
print('(row=0, col=1) index 가리키는 값:', array2d[0,1])
print('(row=1, col=0) index 가리키는 값:', array2d[1,0])
print('(row=2, col=2) index 가리키는 값:', array2d[2,2])


# In[3]:


array1=np.arange(start=1, stop=10)
array3= array1[0:3]
print(array3)
print(type(array3))


# In[4]:


array1=np.arange(start=1, stop=10)
array4=array1[:3]
print(array4)

array5=array1[3:]
print(array5)

array6=array1[:]
print(array6)


# In[8]:


array1d=np.arange(start=1, stop=10)
array2d=array1d.reshape(3,3)
print('array2d:\n', array2d)

print('array2d[0:2],[0:2] \n', array2d[0:2,0:2])
print('array2d[1:3],[0:3] \n', array2d[1:3,0:3])
print('array2d[1:3],[:] \n', array2d[1:3,:])
print('array2d[:],[:] \n', array2d[:,:])
print('array2d[:2],[1:] \n', array2d[:2,1:])
print('array2d[:2],[0] \n', array2d[:2,0])


# In[11]:


print(array2d[0])
print(array2d[1])
print('array2d[0] shape:', array2d[0].shape, 'array2d[1] shape:', array2d[1].shape)


# In[14]:


array1d=np.arange(start=1, stop=10)
array2d=array1d.reshape(3,3)

array3=array2d[[0,1],2]
print('array2d[[0,1],2]=>', array3.tolist())

array4=array2d[[0,1],0:2]
print('array2d[[0,1],0:2]=>', array4.tolist())

array5=array2d[[0,1]]
print('array2d[[0,1]]=>', array5.tolist())


# In[16]:


array1d=np.arange(start=1, stop=10)
#[]안에 array1d > 5 Boolean indexing을 적용
#조건 필터링과 검색 동시 가능
array3 = array1d[array1d>5]
print('array1d>5 불린 인덱싱 결과 값 :', array3)

array1d>5


# In[17]:


boolean_indexes=np.array([False, False, False, False, False,  True,  True,  True,  True])
array3=array1d[boolean_indexes]
print('불린 인덱스로 필터링 결과 :', array3)


# In[18]:


indexes = np.array([5,6,7,8])
array4 = array1d[indexes]
print('일반 인덱스로 필터링 결과 :', array4)


# In[ ]:
org_array = np.array([3,1,9,5])
print('원본 행렬:', org_array)

#np.sort()로 정렬
sort_array1=np.sort(org_array)
print('np.sort() 호출 후 반환된 정렬 행렬', sort_array1)
print('np.sort() 호출 후 원본 행렬', org_array)

#ndarray.sort()로 정렬
sort_array2=org_array.sort()
print('org_array.sort() 호출 후 반환된 정렬 행렬', sort_array2)
print('org_array.sort() 호출 후 원본 행렬', org_array)

# In[ ]:
sort_array1_desc = np.sort(org_array)[::-1]
print('내림차순으로 정렬:', sort_array1_desc)

# In[ ]:
array2d = np.array([[8,12],[7,1]])

sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2d_axis0)

sort_array2d_axis1 = np.sort(array2d, axis=1)
print('칼 방향으로 정렬:\n', sort_array2d_axis1)


# In[ ]:
org_array = np.array([3,1,9,5])
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 행렬의 인덱스:', sort_indices)

# In[ ]:
org_array = np.array([3,1,9,5])
sort_indices_desc = np.argsort(org_array)[::-1]
print('행렬 내림차순 정렬 시 원본 행렬의 인덱스:', sort_indices_desc)

# In[ ]:
import numpy as np

name_array = np.array(['John','Mike','Sarah','Kate','Samuel'])
score_array=np.array([78,95,84,98,88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스:', sort_indices_asc)
print('성적 내림차순 정렬 시 name_array의 이름 출력:', name_array[sort_indices_asc])


# In[ ]:
A = np.array([[1,2,3],
              [4,5,6]])
B = np.array([[7,8],
              [9,10],
              [11,12]])
dot_product = np.dot(A,B)
print('헹렬 내적 결과:\n', dot_product)


# In[ ]:
A = np.array([[1,2],
             [3,4]])
transpose_mat = np.transpose(A)
print('A의 전치행렬;\n', transpose_mat)
