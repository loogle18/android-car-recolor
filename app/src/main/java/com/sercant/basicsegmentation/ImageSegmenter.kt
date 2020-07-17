package com.sercant.basicsegmentation

import android.app.Activity
import android.graphics.Bitmap
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
class ImageSegmenter(
    private val activity: Activity
) {

    companion object {
        const val TAG: String = "ImageSegmenter"
        const val DIM_BATCH_SIZE = 1
        const val DIM_PIXEL_SIZE = 1
        const val DIM_CLASS_SIZE = 2
        const val BYTES_PER_CHANNEL = 4
    }

    /** Pre-allocated buffers for storing image data in.  */
    val xySize = 1024
    val ioSize = xySize * xySize
    val model = Model(
        "lr_aspp_s_new_1024",
        xySize, xySize, xySize, xySize
    )

    /** Options for configuring the Interpreter.  */
    private lateinit var outputPixels: IntArray
    private lateinit var gsPixels: FloatArray
    private lateinit var maskPixels: FloatArray
    private lateinit var segRawValues: FloatArray
    private lateinit var segmentedBitmap: Bitmap
    private val tfliteOptions = Interpreter.Options()

    /** The loaded TensorFlow Lite model.  */
    private var tfliteModel: MappedByteBuffer? = null

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    private var tflite: Interpreter? = null
    private var delegate: GpuDelegate? = null

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.  */
    private lateinit var inputData: ByteBuffer
    private lateinit var segmentedData: ByteBuffer

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

        inputData = ByteBuffer
            .allocateDirect(DIM_BATCH_SIZE * ioSize * DIM_PIXEL_SIZE * BYTES_PER_CHANNEL)
            .order(ByteOrder.nativeOrder())

        segmentedData = ByteBuffer
            .allocateDirect(DIM_CLASS_SIZE * ioSize * BYTES_PER_CHANNEL)
            .order(ByteOrder.nativeOrder())

        outputPixels = IntArray(ioSize)

        gsPixels = FloatArray(ioSize)

        maskPixels = FloatArray(ioSize)

        segRawValues = FloatArray(ioSize * DIM_CLASS_SIZE)

        segmentedBitmap = Bitmap.createBitmap(model.inputWidth, model.inputHeight, Bitmap.Config.ARGB_8888)
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
    private fun convertBitmapToByteBuffer(inputPixels: IntArray) {
        inputData.rewind()

        for (pixel in 0 until ioSize) {
            val value = inputPixels[pixel]
            val newVal = 0.299f * (value shr 16 and 0xff) + 0.587f * (value shr 8 and 0xff) + 0.114f * (value and 0xff)
            gsPixels[pixel] = newVal
        }

        inputData.asFloatBuffer().put(gsPixels)
    }

    private fun rgb(red: Int, green: Int, blue: Int) : Int {
        return 0xff shl 24 or (red shl 16) or (green shl 8) or blue
    }

    /**
     * Segments a frame from the preview stream.
     */

    fun segmentFrame(inputPixels: IntArray, width: Int, height: Int, newColor: Int) : Bitmap {
        // val startP = System.currentTimeMillis()
        convertBitmapToByteBuffer(inputPixels)
        // Log.d("TTTTTTTTTTT", "PRE: ${System.currentTimeMillis() - startP}ms")
        segmentedData.rewind()
        // val startI = System.currentTimeMillis()
        tflite?.run(inputData, segmentedData)
        // Log.d("TTTTTTTTTTT", "INF: ${System.currentTimeMillis() - startI}ms")
        segmentedData.flip()
        segmentedData.asFloatBuffer().get(segRawValues)
        // val startPo = System.currentTimeMillis()

        val gsPixelCounts = IntArray(256) // 0.255 color range
        var mostFreqColor = 0
        var colorFreq = 0

        for (y in 0 until height) {
            for (x in 0 until width) {
                val i = y * width + x
                val xyz = y * height * 2 + x * 2
                val max = segRawValues[xyz]
                val value = segRawValues[xyz + 1]
                if (value > max) {
                    val gsValue = gsPixels[i]
                    val gsValueInt = Math.round(gsValue)
                    val count = gsPixelCounts[gsValueInt] ?: 0
                    val countNew = count + 1
                    gsPixelCounts[gsValueInt] = countNew
                    if (countNew >= colorFreq) {
                        colorFreq = countNew
                        mostFreqColor = gsValueInt
                    }
                    maskPixels[i] = gsValue
                } else {
                    maskPixels[i] = -1.0f
                }
            }
        }

        for (j in 0 until ioSize) {
            val newVal = maskPixels[j]
            if (newVal >= 0) {
                val newColorR = Math.max(Math.min((newColor.red + Math.round(newVal - mostFreqColor)), 255), 0)
                val newColorG = Math.max(Math.min((newColor.green + Math.round(newVal - mostFreqColor)), 255), 0)
                val newColorB = Math.max(Math.min((newColor.blue + Math.round(newVal - mostFreqColor)), 255), 0)
                outputPixels[j] = rgb(newColorR, newColorG, newColorB)
            } else {
                outputPixels[j] = 0x00000000
            }
        }

        segmentedBitmap.setPixels(outputPixels, 0, width, 0, 0, width, height)
        // Log.d("TTTTTTTTTTT", "POS: ${System.currentTimeMillis() - startPo}ms")
        return segmentedBitmap
    }

    data class Model(
        val path: String,
        val inputWidth: Int,
        val inputHeight: Int,
        val outputWidth: Int,
        val outputHeight: Int
    )
}
