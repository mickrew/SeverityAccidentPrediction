import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class IncrementalClassifier {
        public static void main(String[] args) throws Exception {
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File("train.arff"));
            Instances structure = loader.getStructure();
            structure.setClassIndex(0);

            NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
            nb.buildClassifier(structure);

            HoeffdingTree ht = new HoeffdingTree();
            ht.buildClassifier(structure);

            Instance current;
            int count = 0;
            while ((current = loader.getNextInstance(structure)) != null) {
                if (count%10000 == 0)
                    System.out.println(count);
                count++;
                nb.updateClassifier(current);
                ht.updateClassifier(current);
            }

            loader = new ArffLoader();
            loader.setFile(new File("train.arff"));
            Instances instancesTrain = loader.getDataSet();
            instancesTrain.setClassIndex(0);

            Evaluation eval = new Evaluation(instancesTrain);
            eval.crossValidateModel(nb, instancesTrain, 10, new Debug.Random(1));

            Evaluation eval1 = new Evaluation(instancesTrain);
            eval1.crossValidateModel(ht, instancesTrain, 10, new Debug.Random(5));

            System.out.println(eval.toSummaryString("Results Test Naive :\n", false));
            System.out.println(eval.toMatrixString());
            System.out.println(eval.pctCorrect());
            System.out.println(eval.errorRate());

            System.out.println(eval1.toSummaryString("Results Test:\n", false));
            System.out.println(eval1.toMatrixString());
            System.out.println(eval1.pctCorrect());
            System.out.println(eval1.errorRate());
        }

}