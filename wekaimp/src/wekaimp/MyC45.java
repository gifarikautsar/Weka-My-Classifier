/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaimp;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.NoSupportForMissingValuesException;
import weka.core.OptionHandler;
import weka.core.Statistics;
import weka.core.Summarizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 *
 * @author gifarikautsar
 */
public class MyC45 extends Classifier 
    implements OptionHandler, Sourcable, 
             WeightedInstancesHandler, 
             TechnicalInformationHandler {
    private MyC45[] successors;
    private Attribute attribute;
    private double classValue;
    private double[] distribution;
    private Attribute classAttribute;
    
    private Instances m_data;
    private double m_threshold;
    private float confLevel = 0.25f;
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
         // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        makeTree(data);
        prune();
    }
    
    public double classifyInstance(Instance inst) throws NoSupportForMissingValuesException{
        if(inst.hasMissingValue())
            throw new NoSupportForMissingValuesException("C45: no missing values, "
                                                   + "please.");

        if(attribute == null){
            return classValue;
        }
        else{
            return successors[(int) inst.value(attribute)].classifyInstance(inst);
        }
    }
    
    public void makeTree(Instances data){
        m_data = new Instances(data);
        if (data.numInstances() == 0) {
            classAttribute = null;
            classValue = Instance.missingValue();
            distribution = new double[data.numClasses()];
            return;
        }
        double GRs[] = new double[data.numAttributes()];
        for (int i=0; i<data.numAttributes()-1; i++){
            GRs[i] = calculateGainRatio(data, data.attribute(i));
        }
        attribute = data.attribute(Utils.maxIndex(GRs));
        if(GRs[attribute.index()] == 0) {
            attribute = null;
            distribution = new double[data.numClasses()];
            
            for(int i=0; i<data.numInstances(); i++){
                Instance iTemp = (Instance) data.instance(i);
                distribution[(int) iTemp.classValue()]++;
            }
            Utils.normalize(distribution);
            classValue = Utils.maxIndex(distribution);
            classAttribute = data.classAttribute();
        }
        else{
            if(attribute.isNominal()){
                Instances[] splitData = splitData(data, attribute);
                successors = new MyC45[attribute.numValues()];
                for(int i=0; i<attribute.numValues(); i++){
                    successors[i] = new MyC45();
                    successors[i].makeTree(splitData[i]);
                }
            }
            else{
                Instances[] splitData = splitDataNumeric(data, attribute);
                successors = new MyC45[2];
                for(int i=0; i<2; i++){
                    successors[i] = new MyC45();
                    successors[i].makeTree(splitData[i]);
                }
            }
        }
    }
    
    public double calculateGainRatio(Instances data, Attribute att){
        double IG = calculateEntropy(data);
        double IV = 0;
        Instances[] splitData = null;
        int _n;
        if(att.isNominal()){
            splitData = splitData(data, att);
            _n = att.numValues();
        }
        else{
            splitData = splitDataNumeric(data, att);
            _n = 2;
        }
        for(int i = 0; i < _n; i++) {
            if(splitData[i].numInstances() > 0){
                IG -= ((double) splitData[i].numInstances() / (double) data.numInstances())
                        * calculateEntropy(splitData[i]);
                IV -= calculateEntropy(splitData[i]);
            }
        }
        return IG; 
    }
    
    
    public double calculateEntropy(Instances data){
        double[] countClass = new double[data.numClasses()];
        for(int i=0; i<data.numInstances(); i++){
            Instance iTemp = (Instance) data.instance(i);
            countClass[(int) iTemp.classValue()]++;
        }
        double entropy = 0;
        double numData = (double) data.numInstances();
        for(int i=0; i<data.numClasses(); i++){
            if(countClass[i] > 0){
                entropy -= (countClass[i] / numData) * (Utils.log2(countClass[i] / numData));
            }
        }
        return entropy;
    }
    
    private Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
          splitData[i] = new Instances(data, data.numInstances());
        }
        
        for (int i=0; i<data.numInstances(); i++) {
          Instance insTemp = (Instance) data.instance(i);
          splitData[(int) insTemp.value(att)].add(insTemp);
        }
        for (int i = 0; i < splitData.length; i++) {
          splitData[i].compactify();
        }
        return splitData;
    }
    
    private Instances[] splitDataNumeric(Instances data, Attribute att){
        Instances[] splitData = new Instances[2];
        for (int i = 0; i < 2; i++) {
          splitData[i] = new Instances(data, data.numInstances());
        }
        
        double threshold = 0;
        for(int i=0; i<data.numInstances(); i++){
            threshold += (double)data.instance(i).value(att);
        }
        threshold /= (double)data.numInstances();
        m_threshold = threshold;
        
        for(int i=0; i<data.numInstances(); i++){
            Instance insTemp = (Instance) data.instance(i);
            if((double)data.instance(i).value(att) <= threshold)
                splitData[0].add(insTemp);
            else
                splitData[1].add(insTemp);
        }
        for(int i=0; i<2; i++)
            splitData[i].compactify();
        
        return splitData;
    }
    
    public void prune() throws Exception {
        double errorsLargestBranch;
        double errorsLeaf;
        double errorsTree;
        int indexOfLargestBranch;
        MyC45 largestBranch;

        if (successors != null){

            // Prune all subtrees.
            if(attribute.isNominal()){
                for (int i=0;i<attribute.numValues();i++)
                    successors[i].prune();
            }
            else{
                for (int i=0;i<2;i++)
                    successors[i].prune();
            }

            // Compute error for largest branch
            indexOfLargestBranch = indexOfMaxBranch();
//            indexOfLargestBranch = 0;
            
            errorsLargestBranch = successors[indexOfLargestBranch].getEstimatedErrorsForBranch();

            // Compute error if this Tree would be leaf
            errorsLeaf = getEstimatedErrorsForDistribution(new Distribution(m_data));

            // Compute error for the whole subtree
            errorsTree = getEstimatedErrors();

            // Decide if leaf is best choice.
            if (Utils.smOrEq(errorsLeaf,errorsTree+0.1) &&
                Utils.smOrEq(errorsLeaf,errorsLargestBranch+0.1)){

                // Free son Trees
                successors = null;

                distribution = new double[m_data.numClasses()];            
                for(int j=0; j<m_data.numInstances(); j++){
                    Instance iTemp = (Instance) m_data.instance(j);
                    distribution[(int) iTemp.classValue()]++;
                }
                Utils.normalize(distribution);
                classValue = Utils.maxIndex(distribution);
                return;
            }

            // Decide if largest branch is better choice
            // than whole subtree.
            if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)){
                largestBranch = successors[indexOfLargestBranch];
                successors = largestBranch.successors;
                prune();
            }
        }
    }
    
    private double getEstimatedErrors(){
        if(successors == null)
            return 0;
        else{
            double errors = 0;
            for(int i=0; i<successors.length; i++){
                errors += successors[i].getEstimatedErrors();
            }
            return errors;
        }
    }
    
    private double getEstimatedErrorsForBranch() throws Exception{
        if(successors == null)
            return getEstimatedErrorsForDistribution(new Distribution(m_data));
        else{
            Distribution dist = new Distribution(m_data);
            double errors = 0;
            for(int i=0; i<successors.length; i++){
                errors += successors[i].getEstimatedErrorsForBranch();
            }
            return errors;
        }
    }
    
    private double getEstimatedErrorsForDistribution(Distribution dist){
        if (Utils.eq(dist.total(),0))
            return 0;
        else
            return dist.numIncorrect()+
                addErrs(dist.total(), dist.numIncorrect(),confLevel);
    }
    
    private int indexOfMaxBranch() {
        int iMax = -1, max = 0;
        for(int i=0; i<successors.length; i++){
            int n = successors[i].numOfBranch();
            if(max < n){
                max = n; iMax = i;
            }
        }
        
        return 0;
    }
    
    private int numOfBranch(){
        if(successors != null){
            int num = 1;
            for(int i=0; i<successors.length; i++){
                num += successors[i].numOfBranch();
            }            
            return num;
        }
        else
            return 1;
    }
    
    public String toString() {
        if ((distribution == null) && (successors == null)) {
            return "C45: No model built yet.";
        }
        return "C45\n\n" + toString(0);
    }
    
    private String toString(int level){
        StringBuffer printTree = new StringBuffer();
        
        if(attribute == null){
            if(Instance.isMissingValue(classValue)){
                printTree.append(": null");                        
            }
            else{
                printTree.append(": " + classAttribute.value((int) classValue));                
            }
        }
        else{
            if(attribute.isNominal()){
                for(int i=0; i<attribute.numValues(); i++){
                    printTree.append("\n");
                    for(int j=0; j<level; j++){
                        printTree.append("|  ");
                    }
                    printTree.append(attribute.name() + " = " + attribute.value(i));
                    printTree.append(successors[i].toString(level + 1));
                }
            }
            else{
                for(int i=0; i<2; i++){
                    printTree.append("\n");
                    for(int j=0; j<level; j++){
                        printTree.append("|  ");
                    }
                    if(i==0)
                        printTree.append(attribute.name() + " <= " + m_threshold);
                    else
                        printTree.append(attribute.name() + " > " + m_threshold);
                    printTree.append(successors[i].toString(level + 1));
                }
            }
        }
        return printTree.toString();
    }
    
    public static double addErrs(double N, double e, float CF){

        // Check for extreme cases at the low end because the
        // normal approximation won't work
        if (e < 1) {

            // Base case (i.e. e == 0) from documenta Geigy Scientific
            // Tables, 6th edition, page 185
            double base = N * (1 - Math.pow(CF, 1 / N)); 
            if (e == 0) {
                return base; 
          }

            // Use linear interpolation between 0 and 1 like C4.5 does
            return base + e * (addErrs(N, 1, CF) - base);
        }

        // Use linear interpolation at the high end (i.e. between N - 0.5
        // and N) because of the continuity correction
        if (e + 0.5 >= N) {

            // Make sure that we never return anything smaller than zero
            return Math.max(N - e, 0);
        }

        // Get z-score corresponding to CF
        double z = Statistics.normalInverse(1 - CF);

        // Compute upper limit of confidence interval
        double  f = (e + 0.5) / N;
        double r = (f + (z * z) / (2 * N) +
                    z * Math.sqrt((f / N) - 
                                  (f * f / N) + 
                                  (z * z / (4 * N * N)))) /(1 + (z * z) / N);

        return (r * N) - e;
    }
    
    @Override
    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String toSource(String string) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}