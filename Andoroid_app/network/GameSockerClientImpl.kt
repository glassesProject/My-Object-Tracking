// network/GameSocketClient.kt
package com.example.tsukumogami.network

import kotlinx.coroutines.flow.Flow

/**
 * ネットワーク通信クライアントのインターフェース。
 */
interface GameSocketClient {
    fun connect(ip: String, port: Int)
    fun disconnect()
    suspend fun sendMessage(message: String)
    fun getMessages(): Flow<String>
}
