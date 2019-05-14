def convblock(cdim, nb, bits=3):
    L = []

    for k in range(1, bits + 1):
        convname = 'conv' + str(nb) + '_' + str(k)
        L.append(Convolution2D(cdim, 3, 3,
                               border_mode='same',
                               activation='relu',
                               name=convname))

    L.append(MaxPooling2D((2, 2), strides=(2, 2)))

    return L

from keras.models import Sequential
mdl = Sequential()
# Trick :
# dummy-permutation = identity to specify input shape
# index starts at 1 as 0 is the sample dimension
mdl.add( Permute((1,2,3), input_shape=(3,224,224)) )

for l in convblock(64, 1, bits=2):
    mdl.add(l)

for l in convblock(128, 2, bits=2):
    mdl.add(l)

for l in convblock(256, 3, bits=3):
    mdl.add(l)

for l in convblock(512, 4, bits=3):
    mdl.add(l)

for l in convblock(512, 5, bits=3):
    mdl.add(l)

mdl.add( Convolution2D(4096, 7, 7, activation='relu', name='fc6') )
mdl.add( Convolution2D(4096, 1, 1, activation='relu', name='fc7') )
mdl.add( Convolution2D(2622, 1, 1, name='fc8') )
mdl.add( Flatten() )
mdl.add( Activation('softmax') )


from scipy.io import loadmat
data = loadmat('vgg-face.mat',
               matlab_compatible=False,
               struct_as_record=False)
net = data['net'][0,0]
l = net.layers
description = net.classes[0,0].description

