
import java.io.File;
import java.text.DecimalFormat;

import org.encog.ConsoleStatusReportable;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.prune.PruneIncremental;

import static org.encog.persist.EncogDirectoryPersistence.*;


public class TrainGuessWhoANN {

    static String[] outputClasses = { "Alex", "Alfred", "Anita", "Anne", "Bernard" };
    static double[][] encodedNames = { { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1 } };
    // static String[] outputClasses = { /*"Ned Stark", "Kahl Drogo", "Robert Baratheon", "Bronn", "Sansa Stark", "Viserys
    // Targaryen", "Arya Stark", "Cersei Lannister", "Hodor", "Danaerys Targaryen", "Tywin Lannister", "Jaqen H’Ghar",
    // "Littlefinger",*/
    // "Melisandre", "Stannis Baratheon", "Samwell Tarley", "Jon Snow", "Tyrion Lannister", "Joffrey “Baratheon”", "Brienne of
    // Tarth", "Ygritte", "Jaime Lannister", "The Hound", "Robb Stark" };
    // static double[][] encodedNames = { { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }/*,
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
    // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }*/ };



    public static void main(String[] args) {

        String file = "char-dataset.txt";
        String[] fromFile = FileUtil.readFile(file);
        String[] questions = fromFile[0].split(",");
        double[][] dataset = new double[fromFile.length - 1][questions.length];
        double[][] targetOutput = new double[fromFile.length - 1][outputClasses.length];

        createDataset(fromFile, questions, dataset, targetOutput);
        BasicNetwork network = networkSetup(questions);
        MLDataSet trainingSet = new BasicMLDataSet(dataset, targetOutput);
        trainNetwork(trainingSet, network);
        testNetwork(network);
        saveObject(new File("new_network.eg"), network);

        // // define a pattern architecture
        // FeedForwardPattern net_test = new FeedForwardPattern();
        // // Create the Input Layer
        // net_test.setInputNeurons(7);
        // // Output Layer
        // net_test.setOutputNeurons(5);
        // // set activation function
        // net_test.setActivationFunction(new ActivationSigmoid());
        // PruneIncremental prune = new PruneIncremental(trainingSet, net_test, 1000, 1, 10, new ConsoleStatusReportable());
        // // try from 1 to 5 hidden units
        // prune.addHiddenLayer(1, 15);
        // prune.process();
        // BasicNetwork network1 = prune.getBestNetwork();
        // System.out.println("Neural Network created: " + network1.getLayerNeuronCount(0) + "-" + network1.getLayerNeuronCount(1)
        // + "-" + network1.getLayerNeuronCount(2));
    }



    public static void createDataset(String[] fromFile, String[] questions, double[][] dataset, double[][] targetOutput) {

        String[][] datasetWords = new String[fromFile.length - 1][questions.length + 1];
        String[] outputNames = new String[fromFile.length - 1];

        for (int i = 0; i < datasetWords.length; i++) {
            datasetWords[i] = fromFile[i + 1].split(",");
            for (int j = 0; j < questions.length; j++) {
                if (datasetWords[i][j].toLowerCase().equals("yes")) {
                    dataset[i][j] = 1;
                } else {
                    dataset[i][j] = 0;
                }
            }
            outputNames[i] = datasetWords[i][questions.length];
            for (int j = 0; j < outputClasses.length; j++) {
                if (outputNames[i].equals(outputClasses[j])) {
                    for (int k = 0; k < outputClasses.length; k++) {
                        targetOutput[i][k] = encodedNames[j][k];
                    }
                }
            }
        }
    }



    public static BasicNetwork networkSetup(String[] questions) {
        int inputUnits = questions.length;
        int hiddenUnits1 = 4;
        int outputUnits = outputClasses.length;

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, false, inputUnits));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hiddenUnits1));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, outputUnits));

        network.getStructure().finalizeStructure();
        network.reset();

        return network;
    }



    public static void trainNetwork(MLDataSet trainingSet, BasicNetwork ANN) {

        double learningRate = 0.6;
        double momentum = 0.2;
        double desiredError = 0.01;
        double totalError = 1;

        Backpropagation train = new Backpropagation(ANN, trainingSet, learningRate, momentum);

        for (int epoch = 0; totalError > desiredError; epoch++) {
            train.iteration();
            totalError = train.getError();

            if (epoch > 0) {
                System.out.println( "Epoch #" + epoch + " Error: " + totalError);
            }
        }
        train.finishTraining();
    }



    public static void testNetwork(BasicNetwork network) {
        double[] testCase = { 0, 0, 1, 0, 0, 1, 0 };

        MLData data = new BasicMLData(testCase);
        MLData output = network.compute(data);
        double[] normalisedOutput = new double[outputClasses.length];

        System.out.println();

        for (int i = 0; i < outputClasses.length; i++) {
            normalisedOutput[i] = Math.round(output.getData(i));
            System.out.println(output.getData(i));
        }

        for (int i = 0; i < outputClasses.length; i++) {
            int comparisonTotal = 0;
            for (int j = 0; j < outputClasses.length; j++) {
                if (normalisedOutput[j] != encodedNames[i][j]) {
                    comparisonTotal++;
                }
            }
            if (comparisonTotal == 0) {
                DecimalFormat percentage = new DecimalFormat("##.##");
                System.out.println("Is your TEST character " + outputClasses[i] + "?");
                System.out.println("I am " + (percentage.format(output.getData(i) * 100)) + "% sure!!");
            }
        }
    }
}
