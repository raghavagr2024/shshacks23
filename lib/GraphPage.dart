import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

import 'HomePage.dart';
import 'main.dart';


class GraphPage extends StatefulWidget{
  @override
  State<StatefulWidget> createState() {

    return _GraphPage();
  }

}

class _GraphPage extends State<GraphPage>{


  @override
  void initState(){
    print('data');
    print(data.toString());
    super.initState();
  }
  @override
  Widget build(BuildContext context) {
    return SafeArea(child: Scaffold(
      body: SfCartesianChart(
        primaryXAxis: NumericAxis(
          isVisible: false
        ),
        series: <ChartSeries>[
          LineSeries<RatingsData,int>(

            dataSource: data,
              xValueMapper: (RatingsData ratings, _) => ratings.date,
              yValueMapper: (RatingsData ratings, _) => ratings.rating,
          )
        ],
      ),
    ));
  }







}

class RatingsData {
  RatingsData(this.date, this.rating);
  final int date;
  final int rating;
}



