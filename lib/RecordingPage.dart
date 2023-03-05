import 'dart:io';
import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:better_open_file/better_open_file.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shshacks23/main.dart';
import 'package:http/http.dart' as http;
void main() {
  runApp(const CameraAwesomeApp());
}

class CameraAwesomeApp extends StatelessWidget {
  const CameraAwesomeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'camerAwesome',
      home: CameraPage(),
    );
  }
}

class CameraPage extends StatelessWidget {
  const CameraPage({super.key});

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      body: CameraAwesomeBuilder.awesome(
        enableAudio: true,
        sensor: Sensors.front,
        saveConfig: SaveConfig.video(
          pathBuilder: () async {
            final Directory extDir = await getTemporaryDirectory();
            final testDir = await Directory('${extDir.path}/test').create(recursive: true);
            return '${testDir.path}/${DateTime.now().millisecondsSinceEpoch}.mp4';
          },
        ),
        aspectRatio: CameraAspectRatios.ratio_16_9,
        previewFit: CameraPreviewFit.fitWidth,
        onMediaTap: (mediaCapture) {

          OpenFile.open(mediaCapture.filePath);

          storageRef.child(uid).child("${DateTime.now().year}-${DateTime.now().month}-${DateTime.now().day}").putFile(File(mediaCapture.filePath));
          print("uid");
          print(uid);
          http.post(Uri.parse('http://10.215.76.175:5000/video/$uid'));
        },
      ),
    );
  }

  Future<String> _path(CaptureMode captureMode) async {
    final Directory extDir = await getTemporaryDirectory();
    final testDir =
        await Directory('${extDir.path}/test').create(recursive: true);
    final String fileExtension =
        captureMode == CaptureMode.photo ? 'jpg' : 'mp4';
    final String filePath =
        '${testDir.path}/${DateTime.now().millisecondsSinceEpoch}.$fileExtension';
    return filePath;
  }
}
