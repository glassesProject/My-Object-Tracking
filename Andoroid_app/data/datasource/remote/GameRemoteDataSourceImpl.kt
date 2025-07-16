// data/datasource/remote/GameRemoteDataSourceImpl.kt
package com.example.tsukumogami.data.datasource.remote

import com.example.tsukumogami.data.model.ServerMessage
import com.example.tsukumogami.network.GameSocketClient
import com.google.gson.Gson
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject

/**
 * GameRemoteDataSourceの具象クラス。
 * GameSocketClientを使って実際にサーバーと通信し、受け取ったJSON文字列をServerMessageオブジェクトに変換する。
 */
class GameRemoteDataSourceImpl @Inject constructor(
    private val socketClient: GameSocketClient,
    private val gson: Gson
) : GameRemoteDataSource {

    override fun connect(ip: String, port: Int): Flow<ServerMessage> {
        socketClient.connect(ip, port)
        return socketClient.getMessages().map { jsonString ->
            gson.fromJson(jsonString, ServerMessage::class.java)
        }
    }

    override suspend fun sendCommand(command: String) {
        val jsonCommand = gson.toJson(mapOf("command" to command))
        socketClient.sendMessage(jsonCommand)
    }

    override fun disconnect() {
        socketClient.disconnect()
    }
}