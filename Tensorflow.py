import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#Data
x=np.array([[0,0],[0,1],[1,0],[1,1]])
#print(x)
y=np.array([[0],[0],[0],[1]])
print(y)

#Build Architecture
model=keras.Sequential()
model.add(layers.Input(shape=(2,)))#Everu neuron recieves 2 inputs
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

#compile
model.compile( loss='mse' , 
     optimizer=keras.optimizers.SGD(learning_rate=0.1),
     metrics=['accuracy'])

#Train
model.fit(x,
          y,
          epochs=30)
loss = model.evaluate(x,y)
print(loss)
    
    
    
    
    
print("Hello World")