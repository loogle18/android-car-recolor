package com.sercant.basicsegmentation

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.core.graphics.blue
import androidx.core.graphics.green
import androidx.core.graphics.red
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Image segmenter
 *
 * @author sercant
 * @date 05/12/2018
 */
class ImageSegmenterOld(
    private val activity: Activity
) {

    companion object {
        const val TAG: String = "ImageSegmenter"
        const val DIM_BATCH_SIZE = 1
        const val DIM_PIXEL_SIZE = 1
        const val BYTES_PER_CHANNEL = 4
    }

    val model = Model(
        "lr_aspp_s_1024",
        1024, 1024, 1024, 1024
    )

    /** Pre-allocated buffers for storing image data in.  */
    private var intValues = IntArray(0)
    private var gsPixels = FloatArray(model.inputWidth * model.inputHeight)
    private var outFrame = IntArray(0)

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions = Interpreter.Options()

    /** The loaded TensorFlow Lite model.  */
    private var tfliteModel: MappedByteBuffer? = null

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    private var tflite: Interpreter? = null
    private var delegate: GpuDelegate? = null

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.  */
    private lateinit var imgData: ByteBuffer

    private lateinit var segmentedImage: ByteBuffer

    init {
        loadModel()
        Log.d(TAG, "Created a Tensorflow Lite Image Segmenter.")
    }

    private fun loadModel() {
        tfliteModel = loadModelFile(activity)
        recreateInterpreter()
    }

    private fun recreateInterpreter() {
        close()
        tfliteModel?.let {
            delegate = GpuDelegate()
            tfliteOptions.setNumThreads(4).addDelegate(delegate)
            tflite = Interpreter(it, tfliteOptions)
        }
        imgData = ByteBuffer
            .allocateDirect(DIM_BATCH_SIZE * model.inputWidth * model.inputHeight * DIM_PIXEL_SIZE * BYTES_PER_CHANNEL)
            .order(ByteOrder.nativeOrder())

        segmentedImage = ByteBuffer
            .allocateDirect(2 * model.outputWidth * model.outputHeight * BYTES_PER_CHANNEL)
            .order(ByteOrder.nativeOrder())

        outFrame = IntArray(model.outputWidth * model.outputHeight)
        intValues = IntArray(model.inputWidth * model.inputHeight)
    }

    /** Closes tflite to release resources.  */
    private fun close() {
        tflite?.close()
        delegate?.close()
    }

    /** Memory-map the model file in Assets.  */
    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd("${model.path}.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /** Writes Image data into a `ByteBuffer`.  */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        imgData.rewind()

        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixel in 0 until intValues.size) {
            val value = intValues[pixel]
            val newVal = 0.299f * (value shr 16 and 0xff) + 0.587f * (value shr 8 and 0xff) + 0.114f * (value and 0xff)
            gsPixels[pixel] = newVal
        }

        imgData.asFloatBuffer().put(gsPixels)
    }

    /**
     * Segments a frame from the preview stream.
     */

    fun segmentFrame(bitmap: Bitmap, width: Int, height: Int, newColor: Int) : Bitmap {
        // val startP = System.currentTimeMillis()
        convertBitmapToByteBuffer(bitmap)
        // Log.d("TTTTTTTTTTT", "PRE: ${System.currentTimeMillis() - startP}ms")
        segmentedImage.rewind()
        // val startI = System.currentTimeMillis()
        tflite?.run(imgData, segmentedImage)
        // Log.d("TTTTTTTTTTT", "INF: ${System.currentTimeMillis() - startI}ms")
        segmentedImage.flip()
        // val startPo = System.currentTimeMillis()
        val output: Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val maskPixels = arrayOfNulls<Float?>(width * height)
        val map = mutableMapOf<Int, Int>()
        val pixels = IntArray(width * height)
        var mostFreqColor = 0
        var colorFreq = 0
        for (i in 0 until width * height) {
            val max: Float = segmentedImage.float
            val value: Float = segmentedImage.float
            if (value > max && value > 0.5f) {
                val newVal = gsPixels[i]
                val newValInt = Math.round(newVal)
                val count: Int? = map[newValInt]
                var countNew = 1
                if (count != null) {
                    map[newValInt] = count + 1
                    countNew = count + 1
                } else {
                    map[newValInt] = 1
                }
                if (countNew >= colorFreq) {
                    colorFreq = countNew
                    mostFreqColor = newValInt
                }
                maskPixels[i] = newVal
            }
        }

        for (i in 0 until width * height) {
            val newVal: Float? = maskPixels[i]
            if (newVal != null) {
                val newColorR = Math.max(Math.min((newColor.red + Math.round(newVal - mostFreqColor)), 255), 0)
                val newColorG = Math.max(Math.min((newColor.green + Math.round(newVal - mostFreqColor)), 255), 0)
                val newColorB = Math.max(Math.min((newColor.blue + Math.round(newVal - mostFreqColor)), 255), 0)
                pixels[i] = Color.rgb(newColorR, newColorG, newColorB)
            } else {
                pixels[i] = 0x00000000
            }
        }
        output.setPixels(pixels, 0, width, 0, 0, width, height)
        // Log.d("TTTTTTTTTTT", "POS: ${System.currentTimeMillis() - startPo}ms")
        return output
    }

    data class Model(
        val path: String,
        val inputWidth: Int,
        val inputHeight: Int,
        val outputWidth: Int,
        val outputHeight: Int
    )
}
