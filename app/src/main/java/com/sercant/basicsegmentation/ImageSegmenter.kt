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

class ImageSegmenter(private val activity: Activity) {
    companion object {
        const val TAG: String = "ImageSegmenter"
        const val DIM_BATCH_SIZE = 1
        const val DIM_PIXEL_SIZE = 1
        const val DIM_CLASS_SIZE = 2
        const val BYTES_PER_CHANNEL = 4
    }

    val zero = 0
    val xySize = 1024
    val ioSize = xySize * xySize
    val model = Model(
        "lr_aspp_s_1024",
        xySize, xySize, xySize, xySize
    )

    private lateinit var outputPixels: IntArray
    private lateinit var gsPixels: FloatArray
    private lateinit var maskPixels: FloatArray
    private lateinit var segRawValues: FloatArray
    private lateinit var segmentedBitmap: Bitmap
    private lateinit var inputData: ByteBuffer
    private lateinit var segmentedData: ByteBuffer

    private val tfliteOptions = Interpreter.Options()
    private var tfliteModel: MappedByteBuffer? = null
    private var tflite: Interpreter? = null
    private var delegate: GpuDelegate? = null

    private var randomR: Float = 1.0f
    private var randomG: Float = 1.0f
    private var randomB: Float = 1.0f

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
            tfliteOptions.setNumThreads(2).addDelegate(delegate)
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

    private fun close() {
        tflite?.close()
        delegate?.close()
    }

    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd("${model.path}.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun setRandomRGB(newColor: Int) {
        if (newColor.red == newColor.green && newColor.green == newColor.blue) {
            randomR = 1.0f
            randomG = 1.0f
            randomB = 1.0f
        } else {
            randomR = (89..99).random().toFloat() / 100.0f
            randomG = (89..99).random().toFloat() / 100.0f
            randomB = (89..99).random().toFloat() / 100.0f
        }
    }

    private fun convertBitmapToByteBuffer(inputPixels: IntArray) {
        inputData.rewind()

        for (pixel in zero until ioSize) {
            val value = inputPixels[pixel]
            val newVal = 0.299f * (value shr 16 and 0xff) + 0.587f * (value shr 8 and 0xff) + 0.114f * (value and 0xff)
            gsPixels[pixel] = newVal
        }

        inputData.asFloatBuffer().put(gsPixels)
    }

    private fun getNewColorDiff(mostFreqGsColor: Int, origGsColor: Float, newGsColor: Float) : Float {
        val origMostFreqGsDiff = origGsColor - mostFreqGsColor
        val newOrigGsDiff = newGsColor - origGsColor
        var result: Float

        val finalRatio = if (origMostFreqGsDiff == 0.0f) {
            2.0f
        } else if (origMostFreqGsDiff >= -25 && origMostFreqGsDiff <= 25) {
            2.0f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else if (origMostFreqGsDiff >= -50 && origMostFreqGsDiff <= 50) {
            2.1f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else if (origMostFreqGsDiff >= -75 && origMostFreqGsDiff <= 75) {
            2.2f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else if (origMostFreqGsDiff >= -100 && origMostFreqGsDiff <= 100) {
            2.3f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else if (origMostFreqGsDiff >= -125 && origMostFreqGsDiff <= 125) {
            2.4f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else if (origMostFreqGsDiff >= -150 && origMostFreqGsDiff <= 150) {
            2.5f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else if (origMostFreqGsDiff >= -175 && origMostFreqGsDiff <= 175) {
            2.6f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else if (origMostFreqGsDiff >= -200 && origMostFreqGsDiff <= 200) {
            2.7f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else if (origMostFreqGsDiff >= -225 && origMostFreqGsDiff <= 225) {
            2.8f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else if (origMostFreqGsDiff >= -250 && origMostFreqGsDiff <= 250) {
            2.9f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        } else {
            3.0f * (origMostFreqGsDiff / -origMostFreqGsDiff)
        }

        if (newOrigGsDiff == 0.0f) {
            result = origMostFreqGsDiff * (finalRatio - 1)
        } else if (newOrigGsDiff >= -25 && newOrigGsDiff <= 25) {
            result = origMostFreqGsDiff * (finalRatio - 1) * (newOrigGsDiff / -newOrigGsDiff)
        } else if (newOrigGsDiff >= -50 && newOrigGsDiff <= 50) {
            result = origMostFreqGsDiff * (finalRatio - 0.9f) * (newOrigGsDiff / -newOrigGsDiff)
        } else if (newOrigGsDiff >= -75 && newOrigGsDiff <= 75) {
            result = origMostFreqGsDiff * (finalRatio - 0.8f) * (newOrigGsDiff / -newOrigGsDiff)
        } else if (newOrigGsDiff >= -100 && newOrigGsDiff <= 100) {
            result = origMostFreqGsDiff * (finalRatio - 0.7f) * (newOrigGsDiff / -newOrigGsDiff)
        } else if (newOrigGsDiff >= -125 && newOrigGsDiff <= 125) {
            result = origMostFreqGsDiff * (finalRatio - 0.6f) * (newOrigGsDiff / -newOrigGsDiff)
        } else if (newOrigGsDiff >= -150 && newOrigGsDiff <= 150) {
            result = origMostFreqGsDiff * (finalRatio - 0.5f) * (newOrigGsDiff / -newOrigGsDiff)
        } else if (newOrigGsDiff >= -175 && newOrigGsDiff <= 175) {
            result = origMostFreqGsDiff * (finalRatio - 0.4f) * (newOrigGsDiff / -newOrigGsDiff)
        } else if (newOrigGsDiff >= -200 && newOrigGsDiff <= 200) {
            result = origMostFreqGsDiff * (finalRatio - 0.3f) * (newOrigGsDiff / -newOrigGsDiff)
        } else if (newOrigGsDiff >= -225 && newOrigGsDiff <= 225) {
            result = origMostFreqGsDiff * (finalRatio - 0.2f) * (newOrigGsDiff / -newOrigGsDiff)
        } else if (newOrigGsDiff >= -250 && newOrigGsDiff <= 250) {
            result = origMostFreqGsDiff * (finalRatio - 0.1f) * (newOrigGsDiff / -newOrigGsDiff)
        } else {
            result = origMostFreqGsDiff * (finalRatio - 0) * (newOrigGsDiff / -newOrigGsDiff)
        }

        when {
            mostFreqGsColor <= 25 -> {
                result *= 0.45f
            }
            mostFreqGsColor <= 50 -> {
                result *= 0.5f
            }
            mostFreqGsColor <= 75 -> {
                result *= 0.55f
            }
            mostFreqGsColor <= 100 -> {
                result *= 0.6f
            }
            mostFreqGsColor <= 125 -> {
                result *= 0.65f
            }
            mostFreqGsColor <= 150 -> {
                result *= 0.7f
            }
            mostFreqGsColor <= 175 -> {
                result *= 0.75f
            }
            mostFreqGsColor <= 200 -> {
                result *= 0.8f
            }
            mostFreqGsColor <= 225 -> {
                result *= 0.85f
            }
            mostFreqGsColor <= 250 -> {
                result *= 0.9f
            }
            else -> {
                result *= 0.95f
            }
        }

        when {
            newGsColor <= 25 -> {
                result *= 0.7f
            }
            newGsColor <= 50 -> {
                result *= 0.73f
            }
            newGsColor <= 75 -> {
                result *= 0.76f
            }
            newGsColor <= 100 -> {
                result *= 0.79f
            }
            newGsColor <= 125 -> {
                result *= 0.82f
            }
            newGsColor <= 150 -> {
                result *= 0.85f
            }
            newGsColor <= 175 -> {
                result *= 0.88f
            }
            newGsColor <= 200 -> {
                result *= 0.91f
            }
            newGsColor <= 225 -> {
                result *= 0.94f
            }
            newGsColor <= 250 -> {
                result *= 0.97f
            }
            else -> {
                result *= 1.0f
            }
        }

        return result
    }

    private fun rgb(red: Int, green: Int, blue: Int) : Int {
        return 0xff shl 24 or (red shl 16) or (green shl 8) or blue
    }

    fun segmentFrame(inputPixels: IntArray, width: Int, height: Int, newColor: Int) : Bitmap {
        setRandomRGB(newColor)
        convertBitmapToByteBuffer(inputPixels)
        segmentedData.rewind()
        tflite?.run(inputData, segmentedData)
        segmentedData.flip()
        segmentedData.asFloatBuffer().get(segRawValues)

        val gsPixelCounts = IntArray(256) // 0.255 color range
        var mostFreqGsColor = 0
        var colorFreq = 0
        val gsColorNew = 0.299f * newColor.red + 0.587f * newColor.green + 0.114f * newColor.blue

        for (y in zero until height) {
            for (x in zero until width) {
                val i = y * width + x
                val xyz = y * height * 2 + x * 2
                val max = segRawValues[xyz]
                val value = segRawValues[xyz + 1]
                if (value > max) {
                    val gsValue = gsPixels[i]
                    val gsValueInt = Math.round(gsValue)
                    val count = gsPixelCounts[gsValueInt]
                    val countNew = count + 1
                    gsPixelCounts[gsValueInt] = countNew
                    if (countNew >= colorFreq) {
                        colorFreq = countNew
                        mostFreqGsColor = gsValueInt
                    }
                    maskPixels[i] = gsValue
                } else {
                    maskPixels[i] = -1.0f
                }
            }
        }

        for (i in zero until ioSize) {
            val gsColor = maskPixels[i]
            if (gsColor >= 0) {
                val newColorDiff = getNewColorDiff(mostFreqGsColor, gsColor, gsColorNew)
                val newColorR = Math.max(Math.min(((newColor.red + Math.round(newColorDiff * randomR))), 255), 0)
                val newColorG = Math.max(Math.min(((newColor.green + Math.round(newColorDiff * randomG))), 255), 0)
                val newColorB = Math.max(Math.min(((newColor.blue + Math.round(newColorDiff * randomB))), 255), 0)
                outputPixels[i] = rgb(newColorR, newColorG, newColorB)
            } else {
                outputPixels[i] = 0x00000000
            }
        }

        segmentedBitmap.setPixels(outputPixels, 0, width, 0, 0, width, height)
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
