// data/repository/GameRepositoryImpl.kt
package com.example.tsukumogami.data.repository

import com.example.tsukumogami.data.datasource.remote.GameRemoteDataSource
import com.example.tsukumogami.data.model.ServerMessage
import com.example.tsukumogami.util.Constants
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

/**
 * GameRepositoryの具象クラス。
 * どのデータソース（この場合はリモート）からデータを取得するかを決定する。
 */
class GameRepositoryImpl @Inject constructor(
    private val remoteDataSource: GameRemoteDataSource
) : GameRepository {

    override fun connectToServer(): Flow<ServerMessage> {
        // 接続情報は定数ファイルから取得
        return remoteDataSource.connect(Constants.SERVER_IP, Constants.SERVER_PORT)
    }

    override suspend fun sendCommand(command: String) {
        remoteDataSource.sendCommand(command)
    }

    override fun disconnect() {
        remoteDataSource.disconnect()
    }
}
