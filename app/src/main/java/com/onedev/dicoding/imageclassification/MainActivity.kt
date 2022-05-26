package com.onedev.dicoding.imageclassification

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.view.View
import android.widget.Toast
import androidx.annotation.RequiresApi
import com.bumptech.glide.Glide
import com.github.dhaval2404.imagepicker.ImagePicker
import com.onedev.dicoding.imageclassification.databinding.ActivityMainBinding
import com.onedev.dicoding.imageclassification.ml.MobilenetV110224Quant
import com.onedev.dicoding.imageclassification.ml.PlantDiseaseDetection
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity(), View.OnClickListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var bitmap: Bitmap
    private var encodedImage: String? = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnSelectPhoto.setOnClickListener(this)
    }

    private fun selectPhoto() {
        ImagePicker.with(this)
            .crop()
            .compress(1024)
            .start()
    }

    private fun encodeImage(bm: Bitmap): String {
        val byteArrayOutputStream = ByteArrayOutputStream()
        bm.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream)
        val b = byteArrayOutputStream.toByteArray()
        return Base64.encodeToString(b, Base64.DEFAULT)
    }

    @RequiresApi(Build.VERSION_CODES.P)
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (resultCode) {
            Activity.RESULT_OK -> {
                val imageUri = data?.data
                binding.imgSample.setImageURI(imageUri)
                bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imageUri)

                Glide.with(this@MainActivity)
                    .load(imageUri)
                    .into(binding.imgSample)
                encodedImage = encodeImage(bitmap)

                val fileName = "labels_mobilenet_quant_v1_224.txt"
                val inputString = application.assets.open(fileName).bufferedReader().use { it.readText() }
                val townList = inputString.split("\n")

                val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
                val model = MobilenetV110224Quant.newInstance(applicationContext)

                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
                val tBuffer = TensorImage.fromBitmap(resized)
                val byteBuffer = tBuffer.buffer
                inputFeature0.loadBuffer(byteBuffer)

                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer
                val max = getMax(outputFeature0.floatArray)
                binding.tvResult.text = townList[max]

                model.close()
            }
            ImagePicker.RESULT_ERROR -> {
                Toast.makeText(this@MainActivity, "Error: ${ImagePicker.getError(data)}", Toast.LENGTH_SHORT).show()
            }
            else -> {
                Toast.makeText(this@MainActivity, "Dibatalkan", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onClick(v: View?) {
        when (v) {
            binding.btnSelectPhoto -> {
                selectPhoto()
            }
        }
    }

    private fun getMax(arr: FloatArray): Int {
        var ind = 0
        var min = 0.0F

        for (i in 0..1000) {
            if (arr[i] > min) {
                ind = i
                min = arr[i]
            }
        }
        return ind
    }
}