package com.example.tsukumogami.data.datasource.remote

import com.example.tsukumogami.data.model.ServerMessage
import kotlinx.coroutines.flow.Flow

/**
 * リモートデータソース（サーバー通信）のインターフェース。
 * これにより、ViewModelは具体的な通信方法（WebSocketなど）を知る必要がなくなる。
 */
interface GameRemoteDataSource {
    fun connect(ip: String, port: Int): Flow<ServerMessage>
    suspend fun sendCommand(command: String)
    fun disconnect()
}
