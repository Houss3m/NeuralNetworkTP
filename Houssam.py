from math import exp
from random import seed

def combinaison(poids, entree):
    somme = poids[-1]
    for i in range(len(poids)-1):
        somme += poids[i] * entree[i]
    return somme


def transfert(somme):
    return 1.0 / (1.0 + exp(-somme))


def propagation(network, colonne):
    entree = colonne
    for couche in network:
        new_entree = []
        for neuron in couche:
            somme = combinaison(neuron['poids'], entree)
            neuron['output'] = transfert(somme)
            new_entree.append(neuron['output'])
        entree = new_entree
    return entree


def transfert_derivative(output):
    return output * (1.0 - output)


def erreur_de_propagation(network, expected):
    for i in reversed(range(len(network))):
        couche = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(couche)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['poids'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(couche)):
                neuron = couche[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(couche)):
            neuron = couche[j]
            neuron['delta'] = errors[j] * transfert_derivative(neuron['output'])

def miseajours_poids(network, colonne, coef):
    for i in range(len(network)):
        entree = colonne[:-1]
        if i != 0:
            entree = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(entree)):
                neuron['poids'][j] += coef * neuron['delta'] * entree[j]
            neuron['poids'][-1] += coef * neuron['delta']

def train_network(network, train, coef, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for colonne in train:
            outputs = propagation(network, colonne)
            expected = [0 for i in range(n_outputs)]
            expected[colonne[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            erreur_de_propagation(network, expected)
            miseajours_poids(network, colonne, coef)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, coef, sum_error))


def predict(network, row):
    outputs = propagation(network, row)
    return outputs.index(max(outputs))


#training the model
seed(1)
train_dataset = [[0.6,0.1,1],
    [0.2,0.3,0],
    [0.4,0.4,0],
    [0.4,0.2,0],
    [0.5,0.3,1],
    [0.1,0.2,0],
    [0.8,0.7,1]]

training_network = [[{'poids': [0.1, -0.2, 0.1]}, {'poids': [0, 0.2, 0.2]}, {'poids': [0.3, -0.4, 0.5]}],
           [{'poids': [-0.4, 0.1, 0.6, -0.1]}, {'poids': [0.2, -0.1, -0.2, 0.6]}]]


train_network(training_network, train_dataset, 0.5, 500, 2)
for couche in training_network:
    print(couche)

#prediction
prediction_dataset = [[0.3, 0.9],
           [0.7, 0.8],
           [0.6, 0.8],
           [0.2, 0.1]]

new_network = [[{'poids': [4.3, -0.7, -1.3]}, {'poids': [-4.27, 0.5, 1.3]}, {'poids': [-6, 0.5, 2.6]}],
           [{'poids': [-3.6, 2.8, 4.9, -1]}, {'poids': [3.4, -3, -4.8, 1.2]}]]


for row in prediction_dataset:
    prediction = predict(new_network, row)
    if prediction == 0:
        pred = "clou"
    else:
        pred = "vis"
    print (row[0],"\t", row[1],"\t", pred)