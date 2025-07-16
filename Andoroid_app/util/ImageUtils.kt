// util/ImageUtils.kt
package com.example.tsukumogami.util

import android.graphics.BitmapFactory
import android.util.Base64
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap

/**
 * 画像処理に関するヘルパー関数を提供する。
 */
object ImageUtils {
    /**
     * Base64エンコードされた文字列をImageBitmapにデコードする。
     * サーバーから送られてきた画像データをUIで表示できる形式に変換する。
     */
    fun decodeBase64ToBitmap(base64Str: String): ImageBitmap {
        val decodedBytes = Base64.decode(base64Str, Base64.DEFAULT)
        return BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.size).asImageBitmap()
    }
}
