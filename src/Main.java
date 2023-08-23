import moa.classifiers.meta.AdaptiveRandomForest;
import moa.core.Example;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.streams.generators.RandomTreeGenerator;

public class Main {
    public static void main(String[] args) {
        int maxInstancesToProcess = 1000;
        int instancesProcessed = 1;
        int sampleFrequency = 100;

        AdaptiveRandomForest learner = new AdaptiveRandomForest();
        learner.getOptions().setViaCLIString("-s 10"); // 10 learners
        learner.setRandomSeed(5);
        learner.prepareForUse();


        RandomTreeGenerator rtg = new RandomTreeGenerator();
        rtg.getOptions().setViaCLIString("-c 3 -u 10 -o 0"); // 3 classes, 10 numeric features, 0 nominal features
        rtg.prepareForUse();

        BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();
        evaluator.prepareForUse();

        learner.setModelContext(rtg.getHeader());

        while (rtg.hasMoreInstances() &&
                instancesProcessed <= maxInstancesToProcess) {
            Example trainInst = rtg.nextInstance();
            Example testInst = trainInst;

            double[] prediction = learner.getVotesForInstance(testInst);

            evaluator.addResult(testInst, prediction);
            learner.trainOnInstance(trainInst);

            if(instancesProcessed == 1) {
                for(int i = 0 ; i < evaluator.getPerformanceMeasurements().length ; ++i)
                    System.out.print(evaluator.getPerformanceMeasurements()[i].getName() +",");
                System.out.println();
            }

            if(instancesProcessed % sampleFrequency == 0) {
                StringBuilder stb = new StringBuilder();
                for (int i = 0; i < evaluator.getPerformanceMeasurements().length; ++i) {
                    stb.append(evaluator.getPerformanceMeasurements()[i].getValue());
                    stb.append(",");
                }
                System.out.println(stb);
            }
            ++instancesProcessed;
        }
    }
}
