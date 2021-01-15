//package ai.certifai.solution.classification;

//import ai.certifai.Helper;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Random;


public class DriverMonitoringSystem {
    // Define Logger to display specific comment
    private static final Logger Log = org.slf4j.LoggerFactory.getLogger(DriverMonitoringSystem.class);

    // Get the PATH and DOWNLOAD LINK for data and random seed for consistent data random generator
    private static String dataDir;
    private static String downloadLink;

    // Random number generator
    private static final Random rng  = new Random(123);

    // Images are of format given by allowedExtension
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    // Define training parameters
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 2;
    private static int batchSize = 32;

    // Input split for train_set and test_set
    private static InputSplit trainSet, testSet;

    // Transform images for data augmentation
    private static ImageTransform imageTransform;

    // Define label generator
    private static ParentPathLabelGenerator imageLabel = new ParentPathLabelGenerator();

    // Scale input to 0 - 1
    private static DataNormalization imageScaler = new ImagePreProcessingScaler(0,1);

    // Run the class to train through all dataset
    public static void DriverMonitoringSystem() throws IOException {}

    // Make iterator to train through all dataset
    private static DataSetIterator makeIterator(InputSplit imageSplit, boolean trainning) throws IOException{
        ImageRecordReader imageReader = new ImageRecordReader(height, width,channels, imageLabel);

        if(trainning && imageTransform != null)
        {
            imageReader.initialize(imageSplit, imageTransform);
        }
        else
        {
            imageReader.initialize(imageSplit);
        }

        // Iterate all images from record reader for training
        DataSetIterator dataIter = new RecordReaderDataSetIterator(imageReader, batchSize, 1, numClasses);
        dataIter.setPreProcessor(imageScaler);

        return dataIter;
    }

    // Get the train iterator
    public static DataSetIterator trainIterator() throws IOException
    {
        return makeIterator(trainSet, true);
    }

    // Get the test iterator
    public static DataSetIterator testIterator() throws IOException
    {
        return makeIterator(testSet, false);
    }

    // Training setup with augmented images
    public static void setup(int batchSizeArg, int trainPerc, ImageTransform imageTransformNew) throws IOException
    {
        imageTransform = imageTransformNew;
        setup(batchSizeArg,trainPerc);
    }

    // Training setup with images from data source
    public static void setup(int batchSizeArg, int trainPerc) throws IOException
    {
//         Data path System and Helper
            dataDir = Paths.get(
            System.getProperty("user.home"),
            Helper.getPropValues("dl4j_home.data")
    ).toString();
//         dataDir = "C:\\Users\\User\\.deeplearning4j\\data\\";

        // Download link path Helper
        downloadLink = Helper.getPropValues("dataset.drivermonitoring.url");

        // Get file from parent directory
        File parentDir = new File(Paths.get(dataDir,"driver-monitoring-system").toString());
        // Get download link file if parent path file not exist
        if(!parentDir.exists())
        {
            downloadAndUnzip();
        }
        // Get the batchSize from setup
        batchSize = batchSizeArg;

        // Files in directories under the parent dir that have "allowed extensions" and random number generator for splitting into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);

        // Balanced path filter for imageLabel to label random images with allowedExtensions
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, imageLabel);
        // Check valid or invalid train percentage
        if (trainPerc >= 100)
        {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }

        // Split the image files into train and test
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainSet = filesInDirSplit[0];
        testSet = filesInDirSplit[1];
    }

    // DownloadUnzip class
    private static void downloadAndUnzip() throws IOException
    {
        // Get the path and define a zip file variable
        String dataZipPath = new File(dataDir).getAbsolutePath();
        File zipFile = new File(dataZipPath, "driver-monitoring-system.zip");

        // Downloading with log
        Log.info("Downloading the dataset from "+downloadLink+ "...");
        FileUtils.copyURLToFile(new URL(downloadLink), zipFile);

        // Downloaded file imcomplete due to absolute path checksum not correct
        if(!Helper.getCheckSum(zipFile.getAbsolutePath())
                .equalsIgnoreCase(Helper.getPropValues("dataset.drivermonitoringsystem.hash"))){
            Log.info("Downloaded file is incomplete");
            System.exit(0);
        }

        // Unzip download file
//        Log.info("Unzipping "+zipFile.getAbsolutePath());
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataZipPath);
    }
}
