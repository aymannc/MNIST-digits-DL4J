import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class MNISTDigits {

    static String basePath = "C:\\datasets\\mnist_png";
    static int height = 28;
    static int width = 28;
    static int channels = 1;
    static int outputNum = 10;
    static int batchSize = 128;
    static int epochCount = 5;
    static int seed = 98;
    static int labelIndex = 1;
    static double learningRate = 0.001;

    public static void main(String[] args) throws IOException {

        MultiLayerNetwork model;
        try {
            System.out.println("Loading The model");
            model = ModelSerializer.restoreMultiLayerNetwork(new File(basePath + "\\model.zip"));
        } catch (IOException ignored) {
            MNISTDigits mnist = new MNISTDigits();
            model = mnist.buildModel();
        }

        System.out.println("Load training Data");
        Random randomGenNum = new Random(seed);
        File trainDataFile = new File(basePath + "\\training");
        FileSplit trainFileSplit = new FileSplit(trainDataFile, NativeImageLoader.ALLOWED_FORMATS, randomGenNum);
        ParentPathLabelGenerator labelMarker = new ParentPathLabelGenerator();
        ImageRecordReader trainImageRecordReader = new ImageRecordReader(height, width, channels, labelMarker);
        trainImageRecordReader.initialize(trainFileSplit);

        DataSetIterator trainDataSetIterator = new
                RecordReaderDataSetIterator(trainImageRecordReader, batchSize, labelIndex, outputNum);
        DataNormalization scalar = new ImagePreProcessingScaler(0, 1);
        trainDataSetIterator.setPreProcessor(scalar);

        System.out.println("Load testing Data");
        File testDataFile = new File(basePath + "\\testing");
        FileSplit testFileSplit = new FileSplit(testDataFile, NativeImageLoader.ALLOWED_FORMATS, randomGenNum);
        ImageRecordReader testImageRecordReader = new ImageRecordReader(height, width, channels, labelMarker);
        testImageRecordReader.initialize(testFileSplit);

        DataSetIterator testDataSetIterator = new
                RecordReaderDataSetIterator(testImageRecordReader, batchSize, labelIndex, outputNum);
        testDataSetIterator.setPreProcessor(scalar);


        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        System.out.println("Total params:" + model.numParams());
        Evaluation evaluation = model.evaluate(testDataSetIterator);
        double startingAccuracy = evaluation.accuracy();
        System.out.println("Starting accuracy " + startingAccuracy);
        for (int i = 0; i < epochCount; i++) {
            testDataSetIterator.reset();
            trainDataSetIterator.reset();

            System.out.println("Epoch " + (i + 1));
            model.fit(trainDataSetIterator);
            evaluation = model.evaluate(testDataSetIterator);
            System.out.println(evaluation.stats());
            if (evaluation.accuracy() > startingAccuracy) {
                System.out.println("old accuracy " + startingAccuracy);
                System.out.println("new accuracy " + evaluation.accuracy());
                System.out.println("Saving model !");
                ModelSerializer.writeModel(model, new File(basePath + "/model.zip"), true);
            }
        }

    }

    public MultiLayerNetwork buildModel() {
        System.out.println("Building the model...");
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .kernelSize(5, 5)
                        .nIn(channels)//Input image depth
                        .stride(1, 1)
                        .nOut(30)//nbr of kernel outputted images
                        .activation(Activation.RELU).build())
                .layer(1, new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .kernelSize(3, 3)
                        .nOut(80)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .backpropType(BackpropType.Standard)
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        return model;
    }
}

