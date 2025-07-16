// data/model/ServerMessage.kt
package com.example.tsukumogami.data.model

import com.google.gson.annotations.SerializedName

/**
 * Pythonサーバーとやり取りするJSONメッセージのデータクラス。
 * Python側で送信するJSONのキーと一致させる必要がある。
 */
data class ServerMessage(
    @SerializedName("status") val status: String,
    @SerializedName("distance_score") val distanceScore: Float?,
    @SerializedName("image") val image: String?, // Base64エンコードされた画像データ
    @SerializedName("message") val message: String?
)