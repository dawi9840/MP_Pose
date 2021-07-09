package com.example.mp_pose

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.drawable.Drawable
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.res.ResourcesCompat
import com.example.mp_pose.ml.OpenposeSinglenet
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        findViewById()

    }

    override fun onStart() {
        super.onStart()
        btnClickTest()
    }

    override fun onResume(){
        super.onResume()

    }

    override fun onPause(){
        super.onPause()

    }

    override fun onStop(){
        super.onStop()

    }

    private var btnImg: Button?= null
    private var txtImg: TextView?= null
    private var resultImg: ImageView?= null

    private fun findViewById(){
        btnImg = findViewById(R.id.btn)
        txtImg = findViewById(R.id.txt)
        resultImg = findViewById(R.id.imgView)
    }

    @SuppressLint("SetTextI18n")
    private fun btnClickTest(){
        btnImg?.setOnClickListener {
            val drawedImage = ResourcesCompat.getDrawable(resources, R.drawable.ski_224, null)
            val imgBitmap = drawableToBitmap(drawedImage!!)
            val tensorImg = imgBitmap2TensorImg(imgBitmap)
            val outputArray = loadTflite(tensorImg)
            val heatmaps = outputArray.outputFeature3AsTensorBuffer
            val pafs = outputArray.outputFeature2AsTensorBuffer


            println("heatmap: ${heatmaps.floatArray.asList()}")
            // println("heatmap[3]: ${heatmaps.floatArray[3]}")

            println("Input shape: ${tensorImg.tensorBuffer.shape.asList()}")
            println("heatmap shape: ${heatmaps.shape.asList()}")
            println("paf shape: ${pafs.shape.asList()}")

            // TODO: heatmaps.buffer to show a image.
            // val newOutputBitmap = getOutputImage(heatmaps.buffer)
            // println("newOutputBitmap: ${newOutputBitmap.rowBytes}")
            // resultImg?.setImageBitmap(imgBitmap)
            txtImg?.text = "result."
        }
    }

    private fun drawableToBitmap(drawable: Drawable): Bitmap {
        /** Returns a resized bitmap of the drawable image.  */
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        drawable.setBounds(0, 0, canvas.width, canvas.height)
        drawable.draw(canvas)
        return bitmap
    }

    private fun loadTflite(tensor_img:TensorImage): OpenposeSinglenet.Outputs {
        /** Create output objects and run the model. **/
        val model = OpenposeSinglenet.newInstance(this.applicationContext)
        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
        inputFeature0.loadBuffer(tensor_img.buffer)
        return model.process(inputFeature0)
    }

    private fun imgBitmap2TensorImg(bitmap: Bitmap): TensorImage {
        /** Input image(bitmap) convert to tensor image. **/
        // Initialization code
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        // Create a TensorImage object. This creates the tensor of the corresponding
        // tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
        var tensorImage = TensorImage(DataType.UINT8)

        // Analysis code for every frame
        // Preprocess the image
        tensorImage.load(bitmap)
        return imageProcessor.process(tensorImage)
    }

    private fun getOutputImage(output: ByteBuffer): Bitmap {
        /** Convert ByteBuffer to Bitmap. **/
        output.rewind() // Rewind the output buffer after running.
        val outputWidth = 224
        val outputHeight = 224
        val bitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(outputWidth * outputHeight) // Set your expected output's height and width
        for (i in 0 until outputWidth * outputHeight) {
            val a = 0xFF
            val r: Float = output.float * 255.0f
            val g: Float = output.float * 255.0f
            val b: Float = output.float * 255.0f
            pixels[i] = a shl 24 or (r.toInt() shl 16) or (g.toInt() shl 8) or b.toInt()
        }
        bitmap.setPixels(pixels, 0, outputWidth, 0, 0, outputWidth, outputHeight)
        return bitmap
    }

}

