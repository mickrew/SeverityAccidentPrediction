import weka.attributeSelection.*;
import weka.core.Instances;
import weka.filters.Filter;

import java.util.ArrayList;
import java.util.List;

public class AttributeSelection {
    private final Instances train;
    private final Instances test;

    public AttributeSelection(Instances trainingSet, Instances testSet){
        train = trainingSet;
        test = testSet;
    }

    public List<Instances> cfs_BestFirst(String optionsEval, String optionsSearch) throws Exception{
        weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        if(optionsEval != null){
            String[] optEvalArray = weka.core.Utils.splitOptions(optionsEval);
            eval.setOptions(optEvalArray);
        }
        if(optionsSearch != null){
            String[] optSearchArray = weka.core.Utils.splitOptions(optionsSearch);
            search.setOptions(optSearchArray);
        }
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(train);
        Instances newTrain = Filter.useFilter(train, filter);
        Instances newTest = Filter.useFilter(test, filter);
        List<Instances> list = new ArrayList<>();
        list.add(newTrain);
        list.add(newTest);
        return list;
    }

    public List<Instances> cfs_GreedyStepWise(String optionsEval, String optionsSearch) throws Exception{
        weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();

        if(optionsEval != null){
            String[] optEvalArray = weka.core.Utils.splitOptions(optionsEval);
            eval.setOptions(optEvalArray);
        }
        if(optionsSearch != null){
            String[] optSearchArray = weka.core.Utils.splitOptions(optionsSearch);
            search.setOptions(optSearchArray);
        }
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(train);
        Instances newTrain = Filter.useFilter(train, filter);
        Instances newTest = Filter.useFilter(test, filter);
        List<Instances> list = new ArrayList<>();
        list.add(newTrain);
        list.add(newTest);
        return list;
    }

    public List<Instances> InfoGain_Ranker(String optionsEval, String optionsSearch) throws Exception{
        weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();

        if(optionsEval != null){
            String[] optEvalArray = weka.core.Utils.splitOptions(optionsEval);
            eval.setOptions(optEvalArray);
        }
        if(optionsSearch != null){
            String[] optSearchArray = weka.core.Utils.splitOptions(optionsSearch);
            search.setOptions(optSearchArray);
        }

        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(train);
        Instances newTrain = Filter.useFilter(train, filter);
        Instances newTest = Filter.useFilter(test, filter);
        List<Instances> list = new ArrayList<>();
        list.add(newTrain);
        list.add(newTest);
        return list;
    }

    public List<Instances> PCA_Ranker(String optionsEval, String optionsSearch) throws Exception{
        weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
        PrincipalComponents eval = new PrincipalComponents();
        Ranker search = new Ranker();

        if(optionsEval != null){
            String[] optEvalArray = weka.core.Utils.splitOptions(optionsEval);
            eval.setOptions(optEvalArray);
        }
        if(optionsSearch != null){
            String[] optSearchArray = weka.core.Utils.splitOptions(optionsSearch);
            search.setOptions(optSearchArray);
        }

        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(train);
        Instances newTrain = Filter.useFilter(train, filter);
        Instances newTest = Filter.useFilter(test, filter);
        List<Instances> list = new ArrayList<>();
        list.add(newTrain);
        list.add(newTest);
        return list;
    }
}


/*
------------- CfsSubsetEval options
          -M        Treat missing values as a separate value.
          -L        Don't include locally predictive attributes.
          -Z        Precompute the full correlation matrix at the outset, rather than compute correlations lazily (as needed) during the search. Use this in conjuction with parallel processing in order to speed up a backward search.
          -P <int>  The size of the thread pool, for example, the number of cores in the CPU. (default 1)
          -E <int>  The number of threads to use, which should be >= size of thread pool. (default 1)
          -D        Output debugging info.

------------- GreedyStepwise options
         -C                 Use conservative forward search
         -B                 Use a backward search instead of a forward one.
         -P <start set>     Specify a starting set of attributes. Eg. 1,3,5-7.
         -R                 Produce a ranked list of attributes.
         -T <threshold>     Specify a theshold by which attributes may be discarded from the ranking. Use in conjuction with -R
         -N <num to select> Specify number of attributes to select
         -num-slots <int>   The number of execution slots, for example, the number of cores in the CPU. (default 1)
         -D                 Print debugging output

------------- InfoGain options
        -M      treat missing values as a separate value.
        -B      just binarize numeric attributes instead of properly discretizing them.

------------- Ranker options
        -P <start set>  Specify a starting set of attributes. Eg. 1,3,5-7. Any starting attributes specified are ignored during the ranking.
        -T <threshold>  Specify a theshold by which attributes may be discarded from the ranking.
        -N <num to select>  Specify number of attributes to select

------------- BestFirst
        -P <start set>     Specify a starting set of attributes. Eg. 1,3,5-7.
        -D <0 = backward | 1 = forward | 2 = bi-directional>       Direction of search. (default = 1).
        -N <num>    Number of non-improving nodes to consider before terminating search.
        -S <num>    Size of lookup cache for evaluated subsets. Expressed as a multiple of the number of attributes in the data set. (default = 1)

------------- Principal Components Analysis Options
        -C      Center (rather than standardize) the data and compute PCA using the covariance (rather than the correlation) matrix.
        -R      Retain enough PC attributes to account for this proportion of variance in the original data. (default = 0.95)
        -O      Transform through the PC space and back to the original space.
        -A      Maximum number of attributes to include in transformed attribute names. (-1 = include all)
 */