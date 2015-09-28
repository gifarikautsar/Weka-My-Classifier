/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaimp;

import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author gifarikautsar
 */
public class Wekaimp {
    
    private Instances data;
    
    public Wekaimp(){
        data = null;
    }
    
    public void loadFile(String data_address){
        try {
            data = ConverterUtils.DataSource.read(data_address);
            System.out.println("================================");
            System.out.println("============Isi File============");
            System.out.println("================================");
            System.out.println(data.toString() + "\n");            
        } catch (Exception ex) {
            System.out.println("File tidak berhasil di-load");
        }     
    }
    
    public void resample(){
        Random R = new Random();
        data.resample(R);
        System.out.println(data.toString() + "\n");   
    }
    
    public void removeAttribute(int[] idx){
        try{
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(idx);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            System.out.println(data.toString() + "\n");
        } catch (Exception ex){
            System.out.println("Remove attribute gagal");
        }       
    }
    
    public Classifier naiveBayesClassifier(){
        Classifier model = null;
        try {
            data.setClassIndex(data.numAttributes()-1);
            NaiveBayes prob = new NaiveBayes();
            prob.buildClassifier(data);
            model = prob;
            System.out.println(model.toString());
        } catch (Exception ex) {
            System.out.println("Tidak bisa berhasil membuat model NaiveBayes");
        }
        return model;
    }
    
    public Classifier id3Classifier(){
        Classifier model = null;
        try {
            data.setClassIndex(data.numAttributes()-1);
            Id3 tree = new Id3();
            tree.buildClassifier(data);
            model = tree;
            System.out.println(model.toString());
        } catch (Exception ex) {
            System.out.println("Tidak bisa berhasil membuat model NaiveBayes");
        }
        return model;
    }
    
    public void crossValidation(Classifier model){
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, 10, new Random(1));
            System.out.println("================================");
            System.out.println("========Cross Validation========");
            System.out.println("================================");
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        } catch (Exception ex) {
            System.out.println("Tidak bisa melakukan Cross Validation");
        }
    }
    
    public void percentageSplit(Classifier model, double percent){
        try {
            int trainSize = (int) Math.round(data.numInstances() * percent/100);
            int testSize = data.numInstances() - trainSize;
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(model, test);
            System.out.println("================================");
            System.out.println("========Percentage  Split=======");
            System.out.println("================================");
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        } catch (Exception ex) {
            System.out.println("File tidak berhasil di-load");
        }
    }
    
    public void saveModel(String modelname, Classifier model){
        try {
            SerializationHelper.write(modelname, model);
            System.out.println(modelname + " berhasil dibuat\n");
        } catch (Exception ex) {
            System.out.println(modelname + " tidak bisa dibuat\n");
        }
    }
    
    public Classifier loadModel(String modeladdress){
        Classifier model = null;
        try {
            model  = (Classifier) SerializationHelper.read(modeladdress);
            System.out.println(model.toString());
            System.out.println(modeladdress + " berhasil diload\n");
        } catch (Exception ex) {
            System.out.println(modeladdress + " tidak bisa diload\n");
        }
        return model;
    }
    
    public void classify(String data_address, Classifier model){
        try {
            Instances test = ConverterUtils.DataSource.read(data_address);
            test.setClassIndex(test.numAttributes()-1);
            System.out.println("====================================");
            System.out.println("=== Predictions on user test set ===");
            System.out.println("====================================");
            System.out.println("# - actual - predicted - distribution");
            for (int i = 0; i < test.numInstances(); i++) {
                double pred = model.classifyInstance(test.instance(i));
                double[] dist = model.distributionForInstance(test.instance(i));
                System.out.print((i+1) + " - ");
                System.out.print(test.instance(i).toString(test.classIndex()) + " - ");
                System.out.print(test.classAttribute().value((int) pred) + " - ");
                System.out.println(Utils.arrayToString(dist));
            }
            System.out.println("\n");
        } catch (Exception ex) {
            System.out.println("Tidak berhasil memprediksi hasil\n");
        }
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String file = "data/weather.nominal.arff";
        String testfile = "data/weather.nominal.test.arff";
        Wekaimp w = new Wekaimp();
        Scanner scan = new Scanner(System.in);
        Classifier model = null;
        
        //loadfile
        w.loadFile(file); 
        w.resample();
                
        //remove attribute
        System.out.println("Ingin menghapus atribut? (Y/N)");
        String remattr = scan.next();
        if(remattr.equals("Y") || remattr.equals("y")){
            int idx[] = new int[1];
            System.out.print("Index atribut yang akan dihapus: ");
            idx[0] = scan.nextInt();
            w.removeAttribute(idx);
        }
        
        //create model
        System.out.println("Classifier yang akan digunakan:");
        System.out.println("1. Naive Bayes");
        System.out.println("2. Id3");
        int pil = scan.nextInt();
        if(pil == 1){
            model = w.naiveBayesClassifier();
        }
        else{
            model = w.id3Classifier();
        }
        w.crossValidation(model);
        w.percentageSplit(model, 30);
        //saveModel
        System.out.println("Ingin menyimpan model? (Y/N)");
        String savemodel = scan.next();
        if(savemodel.equals("Y") || savemodel.equals("y")){
            System.out.print("Nama file: ");
            String modelname = scan.next();
            modelname += ".model";
            w.saveModel(modelname, model);     
            model = w.loadModel(modelname);
            w.crossValidation(model);
        }
        w.classify(testfile, model);
    }
}
