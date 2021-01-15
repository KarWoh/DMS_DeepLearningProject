/*
 * Copyright (c) 2019 Skymind AI Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//import DriverMonitoringSystem;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.util.darknet.VOCLabels;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.opencv.dnn.DetectionModel;
import org.slf4j.Logger;

import javax.swing.*;
import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public class EditLastLayerOthersFrozen {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.class);
    private static int seed = 123;
    private static final Random randNumGen = new Random(seed);

    //Camera position change between "front" and "back"
    //front camera requires flipping of the image
    private static String cameraPos = "front";

    //swap between camera with 0 -? on the parameter
    //Default is 0
    private static int cameraNum = 0;
    private static Thread thread;

    private static int width = 224;
    private static int height = 224;
    private static int gridWidth = 13;
    private static int gridHeight =13;
    private static int epochs = 1;
    private static int batchSize = 32;
    private static int numClasses = 2;
    private static ComputationGraph model;
//    private static File modelFilename = new File(System.getProperty("user.home"), ".deeplearning4j/generated-models/DMS.zip");

    private static String modelExportDir;

    public static void main(String[] args) throws Exception {
//        System.out.println( modelFilename );
//
//        if (modelFilename.exists()) {
//            //        STEP 2 : Load trained model from previous execution
//            Nd4j.getRandom().setSeed( seed );
//            log.info( "Load model..." );
//            model = ModelSerializer.restoreComputationGraph(modelFilename);
//        } else {
//            log.info("Model not found.");
//        }

        // image augmentation
        ImageTransform horizontalFlip = new FlipImageTransform( 1 );
        ImageTransform cropImage = new CropImageTransform( 25 );
        ImageTransform rotateImage = new RotateImageTransform( randNumGen, 15 );
        ImageTransform showImage = new ShowImageTransform( "Image", 1000 );
        boolean shuffle = false;
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>( horizontalFlip, 0.5 ),
                new Pair<>( rotateImage, 0.5 ),
                new Pair<>( cropImage, 0.3 )
//                ,new Pair<>(showImage,1.0) //uncomment this to show transform image
        );
        ImageTransform transform = new PipelineImageTransform( pipeline, shuffle );

        //create iterators
        DriverMonitoringSystem.setup( batchSize, 80, transform );
        //create iterators
        DataSetIterator trainIter = DriverMonitoringSystem.trainIterator();
        DataSetIterator testIter = DriverMonitoringSystem.testIterator();

        //load vgg16 zoo model
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();

        // Override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed( seed )
                .updater( new Adam( 0.001, 0.9, 0.98, 10 - 7 ) )
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder( vgg16 )
                .fineTuneConfiguration( fineTuneConf )
                .setFeatureExtractor( "fc2" ) //the specified layer and below are "frozen"
                .removeVertexKeepConnections( "predictions" ) //replace the functionality of the final vertex
                .addLayer( "predictions",
                        new OutputLayer.Builder( LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD )
                                .nIn( 4096 ).nOut( numClasses )
                                .weightInit( WeightInit.XAVIER )
                                .activation( Activation.SOFTMAX ).build(),
                        "fc2" )
                .build();
        log.info( vgg16Transfer.summary() );

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage( new File( System.getProperty( "java.io.tmpdir" ), "ui-stats.dl4j" ) );
        uiServer.attach( statsStorage );
        vgg16Transfer.setListeners(
                new StatsListener( statsStorage ),
                new ScoreIterationListener( 5 ),
                new EvaluativeListener( trainIter, 1, InvocationType.EPOCH_END ),
                new EvaluativeListener( testIter, 1, InvocationType.EPOCH_END )
        );

        vgg16Transfer.fit( trainIter, epochs );
        vgg16Transfer.fit( testIter, epochs );

        Evaluation trainEval = vgg16Transfer.evaluate( trainIter );
        Evaluation testEval = vgg16Transfer.evaluate( testIter );

        System.out.println( trainEval.stats() );
        System.out.println( testEval.stats() );


        modelExportDir = Paths.get(
                System.getProperty( "user.home" ),
                Helper.getPropValues( "dl4j_home.generated-models" )
        ).toString();


        File locationToSaveModel = new File( Paths.get( modelExportDir, "DMS.zip" ).toString() );
        if (!locationToSaveModel.exists()) {
            locationToSaveModel.getParentFile().mkdirs();
        }

        boolean saveUpdater = false;
        ModelSerializer.writeModel( vgg16Transfer, locationToSaveModel, saveUpdater );
        log.info( "Model saved" );
//
//
        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
            throw new Exception("Unknown argument for camera position. Choose between front and back");
        }

        FrameGrabber grabber = FrameGrabber.createDefault(cameraNum);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        grabber.start();
        String winName = "Driver Drowsiness Detection";
        CanvasFrame canvas = new CanvasFrame(winName);
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w, h);

//        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();
        NativeImageLoader loader = new NativeImageLoader(width, height, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        VOCLabels labels = new VOCLabels();

        while (true) {
            Frame frame = grabber.grab();

            //if a thread is null, create new thread
            if (thread == null) {
                thread = new Thread(() ->
                {
                    while (frame != null) {
                        try {
                            Mat rawImage = new Mat();
                            //Flip the camera if opening front camera
                            if (cameraPos.equals("front")) {
                                Mat inputImage = converter.convert(frame);
                                flip(inputImage, rawImage, 1);
                            } else {
                                rawImage = converter.convert(frame);
                            }
                            Mat resizeImage = new Mat();
                            resize(rawImage, resizeImage, new Size(width, height));
                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
//                              outputs = initializedModel.outputSingle(inputImage);

//                            vgg16Transfer.output(inputImage);
                            model.output(inputImage);

//                            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) vgg16Transfer.getOutputLayer( 0 );
//                            List<DetectedObject> objs = yout.getPredictedObjects( outputs, detectionThreshold );
//                            YoloUtils.nms( objs, 0.4 );

//                            for (INDArray obj : model.output(inputImage)) {
//                                double[] xy1;
//                                xy1 = obj.toDoubleVector();
//                                double[] xy2;
//                                xy2 = obj.toDoubleVector();
//                                String label = labels.getLabel(obj.getInt());
//                                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
//                                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
//                                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
//                                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
//                                rectangle(rawImage, new Point(x1, y1), new Point(x2, y2), Scalar.RED, 2, 0, 0);
//                                putText(rawImage, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
//                            }
                            canvas.showImage(converter.convert(rawImage));
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }
            KeyEvent t = canvas.waitKey(33);
            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
        canvas.dispose();
    }
}
