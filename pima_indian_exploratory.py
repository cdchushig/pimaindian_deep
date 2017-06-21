import numpy

# Load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

pregnant_mean = numpy.mean(dataset[:,0])
pregnant_desviation = numpy.std(dataset[:,0])

plasma_mean = numpy.mean(dataset[:,1])
plasma_desviation = numpy.std(dataset[:,1])

triceps_mean = numpy.mean(dataset[:,3])
triceps_desviation = numpy.std(dataset[:,3])


print pregnant_mean
print pregnant_desviation
print plasma_mean
print plasma_desviation
print triceps_mean
print triceps_desviation
