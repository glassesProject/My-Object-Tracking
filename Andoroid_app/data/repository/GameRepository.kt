// data/repository/GameRepository.kt
package com.example.tsukumogami.data.repository

import com.example.tsukumogami.data.model.ServerMessage
import kotlinx.coroutines.flow.Flow

/**
 * データ層へのアクセスを抽象化するリポジトリのインターフェース。
 */
interface GameRepository {
    fun connectToServer(): Flow<ServerMessage>
    suspend fun sendCommand(command: String)
    fun disconnect()
}
