
import static org.encog.persist.EncogDirectoryPersistence.*;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Scanner;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;


public class ANNPlaysGuessWho {

    static String[] outputClasses = null;
    static double[][] encodedNames = null;



    public static void main(String[] args) {

        System.out.println("Would you like to play with the BASIC character/feature set? (Please enter \"1\")");
        System.out.println("Or would you rather play the EXTENDED character/feature set? (Please enter \"2\")");
        Scanner scan = new Scanner(System.in);
        String gameType = scan.nextLine();
        String savedNetwork = null;
        String dataFile = null;
        double[] usersAnswers = null;
        double[] usersAnswers1 = { 0.257, 0.371, 0.371, 0.229, 0.343, 0.486, 0.343 };
        double[] usersAnswers2 = { 0.708333333333333, 0.472222222222222, 0.791666666666667, 0.402777777777778, 0.513888888888889, 0.208333333333333, 0.305555555555556, 0.486111111111111, 0.375, 0.125, 0.138888888888889, 0.125, 0.083333333333333,
                0.166666666666667 };

        if (gameType.equals("1")) {
            savedNetwork = "trained_network.eg";
            usersAnswers = usersAnswers1.clone();
            dataFile = "char-dataset.txt";
            setEncoding(0);
        } else if (gameType.equals("2")) {
            // https://www.tes.com/teaching-resource/gradient-top-trumps-game-of-thrones-6421860
            savedNetwork = "GOT_network.eg";
            usersAnswers = usersAnswers2.clone();
            dataFile = "GOT_training_set.txt";
            setEncoding(1);
        }

        BasicNetwork network = (BasicNetwork) loadObject(new File(savedNetwork));
        String[] fromFile = FileUtil.readFile(dataFile);
        String[] questions = fromFile[0].split(",");
        boolean stillPlaying = true;

        System.out.println();
        System.out.println("For the following questions, please type: \"Yes\" or \"No\", or \"Y\" or \"N\"  -  (Not case sensitive)");

        while (stillPlaying == true) {
            for (int i = 0; i < questions.length; i++) {
                usersAnswers[i] = askQuestions(i, questions);

                if (i >= 3 && i < questions.length - 1) {
                    if (calculateEarlyGuess(network, usersAnswers)) {
                        break;
                    }
                }
            }
            calculateGuess(network, usersAnswers);
            String amICorrect = scan.nextLine().toLowerCase();
            if (amICorrect.equals("yes") || amICorrect.equals("y")) {
                System.out.println("Oh fantastic! I'm basically psychic.");
            } else if (amICorrect.equals("no") || amICorrect.equals("n")) {
                System.out.println("Oh man, please forgive me!");
                System.out.println("We all make mistakes!");
                System.out.println("I'm only human after all...");
            }
            System.out.println("Would you like another game? :)");
            String newGame = scan.nextLine().toLowerCase();
            if (newGame.equals("yes") || newGame.equals("y")) {
                stillPlaying = true;
            } else if (newGame.equals("no") || newGame.equals("n")) {
                stillPlaying = false;
            }
        }
    }



    public static double askQuestions(int featureIndex, String[] questions) {

        double binaryAnswer = -1;
        boolean validAnswer = false;

        while (validAnswer == false) {

            System.out.println("Does your character " + questions[featureIndex]);
            Scanner readIn = new Scanner(System.in);
            String wordAnswer = readIn.nextLine().toLowerCase();

            if (wordAnswer.equals("yes") || wordAnswer.equals("y")) {
                binaryAnswer = 1;
                validAnswer = true;
            } else if (wordAnswer.equals("no") || wordAnswer.equals("n")) {
                binaryAnswer = 0;
                validAnswer = true;
            } else {
                System.out.println("You did not provide a valid answer..... Please take 'Guess Who' seriously, and try again!");
                System.out.println();
            }
        }
        return binaryAnswer;
    }



    public static void calculateGuess(BasicNetwork network, double[] usersAnswers) {

        MLData data = new BasicMLData(usersAnswers);
        MLData output = network.compute(data);
        double[] normalisedOutput = new double[outputClasses.length];

        System.out.println();

        for (int i = 0; i < outputClasses.length; i++) {
            normalisedOutput[i] = Math.round(output.getData(i));
        }
        int comparisonTotal = 0;
        int person = 0;
        int bestAnswer = outputClasses.length;
        for (int i = 0; i < outputClasses.length; i++) {
            comparisonTotal = 0;
            person = 0;
            for (int j = 0; j < outputClasses.length; j++) {
                if (normalisedOutput[j] != encodedNames[i][j]) {
                    comparisonTotal++;
                }
            }
            if (comparisonTotal == 0) {
                person = i;
                break;
            } else if (comparisonTotal < bestAnswer) {
                person = i;
            }
            // System.out.println(output.getData(i));
            // System.out.println(bestAnswer);
        }
        DecimalFormat percentage = new DecimalFormat("##.##");
        System.out.println("Is your character..... " + outputClasses[person] + "?");
        System.out.println("I am " + (percentage.format(output.getData(person) * 100)) + "% sure!!");
        System.out.println("Am I correct??");
    }



    public static boolean calculateEarlyGuess(BasicNetwork network, double[] usersAnswers) {

        MLData data = new BasicMLData(usersAnswers);
        MLData output = network.compute(data);
        boolean highChance = false;
        double guessingThreshold = 0.82;
        System.out.println();

        for (int i = 0; i < outputClasses.length; i++) {
            if (output.getData(i) > guessingThreshold) {
                highChance = true;
            }
        }
        if (highChance == true) {
            System.out.println("I think I might know. May I attempt an early guess?");
            Scanner readIn = new Scanner(System.in);
            String allowed = readIn.nextLine().toLowerCase();
            if (allowed.equals("yes") || allowed.equals("y")) {
                return true;
            } else if (allowed.equals("no") || allowed.equals("n")) {
                return false;
            } else {
                System.out.println("I'll take that as a no then! :/");
                System.out.println();
                return false;
            }

        } else {
            return false;
        }
    }



    public static void setEncoding(int x) {

        if (x == 0) {
            String[] outputClasses0 = { "Alex", "Alfred", "Anita", "Anne", "Bernard" };
            double[][] encodedNames0 = { { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1 } };
            outputClasses = outputClasses0.clone();
            encodedNames = encodedNames0.clone();
        } else if (x == 1) {

            String[] outputClasses1 = { "Ned Stark", "Kahl Drogo", "Robert Baratheon", "Bronn", "Sansa Stark", "Viserys Targaryen", "Arya Stark", "Cersei 0L annister", "Hodor", "Danaerys Targaryen", "Tywin Lannister", "Jaqen H’Ghar", "Littlefinger",
                    "Melisandre", "Stannis Baratheon", "Samwell Tarley", "Jon Snow", "Tyrion Lannister", "Joffrey “Baratheon”", "Brienne ofTarth", "Ygritte", "Jaime Lannister", "The Hound", "Robb Stark" };

            double[][] encodedNames1 = { { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                                            0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0,
                                                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                            0, 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                    1 } };
            outputClasses = outputClasses1.clone();
            encodedNames = encodedNames1.clone();
        }
    }
}
