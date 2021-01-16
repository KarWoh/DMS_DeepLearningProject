import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.event.KeyEvent;
import java.io.File;

import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class WebCamInference {
    //Camera position change between "front" and "back"
    //front camera requires flipping of the image
    private static String cameraPos = "front";

    //swap between camera with 0 -? on the parameter
    //Default is 0
    private static int cameraNum = 0;
    private static Thread thread;
    private static File modelFilename = new File("C:\\Users\\User\\.deeplearning4j\\generated-models\\DMS_VGG16.zip");

    public static void main(String[] args) throws Exception {
        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
            throw new Exception("Unknown argument for camera position. Choose between front and back");
        }

        FrameGrabber grabber = FrameGrabber.createDefault(cameraNum);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        grabber.start();
        String winName = "Drowsiness detection";
        CanvasFrame canvas = new CanvasFrame(winName);
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w, h);

        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        ComputationGraph model = ModelSerializer.restoreComputationGraph(modelFilename);


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

                            INDArray inputImage = loader.asMatrix(rawImage);
                            scaler.transform(inputImage);


                            INDArray[] output = model.output(inputImage);
                            float labelRaw = Nd4j.argMax(output[0], 1).getFloat(0);

                            if (labelRaw == 1.0) {
                                System.out.println("Label: Normal" );
                                System.out.println("Probabilities: " + output[0].toString());
                                putText(rawImage, "Normal", new Point(1, 25), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
                            } else {
                                System.out.println("Label: Sleepy" );
                                System.out.println("Probabilities: " + output[0].toString());
                                putText(rawImage, "Sleepy", new Point(1, 25), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
                            }

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
