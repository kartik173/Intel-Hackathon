from numpy import float32, float64
import numpy as np
import warnings
from daal.algorithms.svm import training, prediction
from daal.algorithms import kernel_function, classifier, multi_class_classifier
from daal.data_management import (BlockDescriptor_Float64, BlockDescriptor, readOnly, readWrite, InputDataArchive, OutputDataArchive, Compressor_Zlib, Decompressor_Zlib,
								  level9, DecompressionStream, CompressionStream, HomogenNumericTable)
from daal.algorithms.multi_class_classifier import quality_metric_set as multiclass_quality
from daal.algorithms.classifier.quality_metric import multiclass_confusion_matrix 
from daal.algorithms.svm import quality_metric_set as twoclass_quality
from daal.algorithms.classifier.quality_metric import binary_confusion_matrix


from collections import namedtuple


# Two-class quality metrics type
TwoClassMetrics = namedtuple('TwoClassMetrics',
		['accuracy', 'precision', 'recall', 'fscore', 'specificity', 'auc'])

# Multi-class quality metrics type
MultiClassMetrics = namedtuple('MultiClassMetrics',
		['averageAccuracy', 'errorRate', 'microPrecision', 'microRecall',
		 'microFscore', 'macroPrecision', 'macroRecall', 'macroFscore'])
		 
class ClassifierQualityMetrics:


	def __init__(self, truth, predictions, nclasses = 2):
		"""Initialize class parameters

		Args:
		   truth: ground truth
		   predictions: predicted labels
		   nclasses: number of classes
		"""

		self._truth = truth
		self._predictions = predictions
		if nclasses == 2:
			self._computeTwoclassQualityMetrics()			
		elif nclasses > 2:
			self._computeMulticlassQualityMetrics(nclasses)
		else:
			raise ValueError('nclasses must be at least 2')



	def get(self, metric):
		"""Get a metric from the quality metrics collection

		Args:
		   metric: name of the metric to return

		Returns:
		   A numeric value for the given metric
		"""
		if metric is not 'confusionMatrix':
			return getattr(self._metrics, metric)
		else:
			return self._confMat
			
	def getAllMetrics(self):
		return self._metrics

	def _computeTwoclassQualityMetrics(self):
		# Alg object for quality metrics computation
		quality_alg = twoclass_quality.Batch()
		# Get access to the input parameter
		input = quality_alg.getInputDataCollection().getInput(
				twoclass_quality.confusionMatrix)
		# Pass ground truth and predictions as input
		input.set(binary_confusion_matrix.groundTruthLabels, self._truth)
		input.set(binary_confusion_matrix.predictedLabels, self._predictions)
		# Compute confusion matrix
		confusion = quality_alg.compute().getResult(twoclass_quality.confusionMatrix)
		#confusion matrix
		self._confMat = confusion.get(binary_confusion_matrix.confusionMatrix)
		# Retrieve quality metrics from the confusion matrix		
		metrics = confusion.get(binary_confusion_matrix.binaryMetrics)
		# Convert the metrics into a Python namedtuple and return it
		block = BlockDescriptor_Float64()
		metrics.getBlockOfRows(0, 1, readOnly, block)
		x = block.getArray().flatten()
		self._metrics = TwoClassMetrics(*x)
		metrics.releaseBlockOfRows(block)



	def _computeMulticlassQualityMetrics(self, nclasses):
		# Alg object for quality metrics computation
		quality_alg = multiclass_quality.Batch(nclasses)
		# Get access to the input parameter
		input = quality_alg.getInputDataCollection().getInput(
				multiclass_quality.confusionMatrix)
		# Pass ground truth and predictions as input
		input.set(multiclass_confusion_matrix.groundTruthLabels, self._truth)
		input.set(multiclass_confusion_matrix.predictedLabels, self._predictions)
		# Compute confusion matrix
		confusion = quality_alg.compute().getResult(multiclass_quality.confusionMatrix)
		#confusion Matrix
		self._confMat = confusion.get(multiclass_confusion_matrix.confusionMatrix)
		# Retrieve quality metrics from the confusion matrix
		metrics = confusion.get(multiclass_confusion_matrix.multiClassMetrics)
		# Convert the metrics into a Python namedtuple and return it
		block = BlockDescriptor_Float64()
		metrics.getBlockOfRows(0, 1, readOnly, block)
		x = block.getArray().flatten()
		self._metrics = MultiClassMetrics(*x)
		metrics.releaseBlockOfRows(block)		 


class MultiSVM:
	'''
	Constructor to set SVM training parameters
	'''
	def __init__(self, nClasses, method="boser", C = 1, tolerence = 0.001, tau = 0.000001, maxIterations = 1000000, cacheSize = 8000000, doShrinking = True, kernel = 'linear',
				 sigma = 0,k=1, b=0,dtype=float64):
		'''
		nClasses: number of classes
		method: 'boser', default: 'boser'
			computation method
		C: deafult: 1
			Upper bound in conditions of the quadratic optimization problem.
		tolerance: default: '0.001'
			Training accuracy/ stopping criteria
		tau: default: 0.000001
			Tau parameter of the WSS scheme.
		maxiterations: default: 1000000
			Maximal number of iterations for the algorithm.
		cacheSize: default: 8000000
			cachesize for storing values of kernal matrix.
		doShringing: True/false, default: True
			flag to set shrinking optimization techinique
		kernel: 'linear'/'rbf', default: 'linear
		k: default: 1
			coefficient value of k when kernal function is 'linear'
		b: 	default: 0
			coeffiecent value of b of linear function
		dtype: intc, float32, float63, intc	
		'''	
		self.method = method
		# Print error message here"
		self.dtype = dtype
		self.C = C
		self.tolerence = tolerence
		self.tau = tau
		self.maxIterations = maxIterations
		self.cachesize = cacheSize
		self.doShrinking = doShrinking
		self.kernel = kernel
		if self.kernel == "rbf":
			self.sigma = sigma
		elif self.kernel == "linear":
			self.k = k
			self.b = b
		self.classes = nClasses
	'''
	Arguments: train data feature values of type nT, train data target values(type nT)
	Returns training results object
	'''
	def training(self, trainData, trainDependentVariables):

		#Set algorithms parameters
		if self.method == 'boser':
			from  daal.algorithms.svm.training import boser
			trainingBatch = training.Batch (method=boser, fptype=self.dtype)
			predictionBatch = prediction.Batch()
	 
		if self.kernel == 'linear':
			trainingBatch.parameter.kernel = kernel_function.linear.Batch (method=boser, fptype=self.dtype)
			trainingBatch.parameter.k = self.k
			trainingBatch.parameter.b = self.b
		elif self.kernel == 'rbf':
			trainingBatch.parameter.kernel =  kernel_function.rbf.Batch(method=boser, fptype=self.dtype)
			trainingBatch.parameter.sigma = self.sigma

		trainingBatch.parameter.cacheSize = self.cachesize
		trainingBatch.parameter.C = self.C
		trainingBatch.parameter.accuracyThreshold = self.tolerence
		trainingBatch.parameter.tau = self.tau
		trainingBatch.parameter.maxIterations = self.maxIterations
		trainingBatch.parameter.doShrinking = self.doShrinking

		algorithm = multi_class_classifier.training.Batch (self.classes)
		algorithm.parameter.training = trainingBatch
		algorithm.parameter.prediction = predictionBatch
		algorithm.input.set (classifier.training.data, trainData)
		algorithm.input.set (classifier.training.labels, trainDependentVariables)
		trainingResult = algorithm.compute ()
		return trainingResult
	'''
	Arguments: training result object, test data feature values(type nT)
	Returns predicted values(type nT)
	'''
	def predict(self, trainingResult, testData): #give other parameters

		if self.method == 'boser':
			from  daal.algorithms.svm.training import boser
			predictionBatch = prediction.Batch (method=boser, fptype=self.dtype)
			trainingBatch = training.Batch (method=boser, fptype=self.dtype)

		if self.kernel == 'linear':
			predictionBatch.parameter.kernel = kernel_function.linear.Batch ()
		elif self.kernel == 'rbf':
			predictionBatch.parameter.kernel = kernel_function.rbf.Batch ()

		algorithm = multi_class_classifier.prediction.Batch (self.classes)
		algorithm.parameter.training = trainingBatch
		algorithm.parameter.prediction = predictionBatch
		algorithm.input.setTable (classifier.prediction.data, testData)
		algorithm.input.setModel (classifier.prediction.model, trainingResult.get (classifier.training.model))
		algorithm.compute ()
		predictionResult = algorithm.getResult()
		predictedResponses = predictionResult.get (classifier.prediction.prediction)
		# Change the predicted values to 1 and -1
		return predictedResponses
	'''
	Arguments: deserialized numpy array
	Returns decompressed numpy array
	'''
	
	def qualityMetrics(self, predictResults, testGroundTruth):
		self._qualityMetricSetResult = ClassifierQualityMetrics( testGroundTruth, predictResults, self.classes)
		return self._qualityMetricSetResult		
		