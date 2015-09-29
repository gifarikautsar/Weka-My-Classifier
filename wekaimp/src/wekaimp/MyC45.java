/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaimp;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 *
 * @author gifarikautsar
 */
public class MyC45 extends Classifier 
    implements TechnicalInformationHandler, Sourcable {
    private MyC45[] successors;
    private Attribute attribute;
    private double classValue;
    private double[] distribution;
    private Attribute classAttribute;
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
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

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
    }
    
    public double classifyInstance(Instance inst) throws NoSupportForMissingValuesException{
        if(inst.hasMissingValue())
            throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                                   + "please.");

        if(attribute == null){
            return classValue;
        }
        else{
            return successors[(int) inst.value(attribute)].classifyInstance(inst);
        }
    }
    
    public void makeTree(Instances data){
        if (data.numInstances() == 0) {
            classAttribute = null;
            classValue = Instance.missingValue();
            distribution = new double[data.numClasses()];
            return;
        }
        double GRs[] = new double[data.numAttributes()];
        for (int i=0; i<data.numAttributes(); i++){
            GRs[i] = calculateGainRatio(data, data.attribute(i));
        }
        classAttribute = data.attribute(Utils.maxIndex(GRs));
        System.out.println(classAttribute.toString());
        if(attribute.index() == 0) {
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
            Instances[] splitData = splitData(data, attribute);
            successors = new MyC45[attribute.numValues()];
            for(int i=0; i<attribute.numValues(); i++){
                successors[i] = new MyC45();
                successors[i].makeTree(splitData[i]);
            }
        }
    }
    
    public double calculateGainRatio(Instances data, Attribute att){
        double IG = calculateEntropy(data);
        double IV = 0;
        Instances[] splitData = splitData(data, att);
        for(int i = 0; i < att.numValues(); i++) {
            if(splitData[i].numInstances() > 0){
                IG -= ((double) splitData[i].numInstances() / (double) data.numInstances())
                        * calculateEntropy(splitData[i]);
                IV -= calculateEntropy(splitData[i]);
            }
        }
        return IG/IV; 
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
            entropy -= (countClass[i] / numData) * (Utils.log2(countClass[i] / numData));
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
            for(int i=0; i<attribute.numValues(); i++){
                printTree.append("\n");
                for(int j=0; j<level; j++){
                    printTree.append("|  ");
                }
                printTree.append(attribute.name() + " = " + attribute.value(i));
                printTree.append(successors[i].toString(level + 1));
            }
        }
        return printTree.toString();
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