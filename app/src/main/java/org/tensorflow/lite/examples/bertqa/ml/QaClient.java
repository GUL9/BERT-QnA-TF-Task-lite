/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.tensorflow.lite.examples.bertqa.ml;
import android.content.Context;
import android.util.Log;
import androidx.annotation.WorkerThread;
import java.io.IOException;
import java.util.List;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.task.text.qa.BertQuestionAnswerer;

/** Interface to load TfLite model and provide predictions. */
public class QaClient {
  private static final String TAG = "BertDemo";
  private static final String MODEL_PATH = "model.tflite";
  private static final int NUM_LITE_THREADS = 4;

  private final Context context;

  private BertQuestionAnswerer model  = null;


  public QaClient(Context context) {
    this.context = context;
  }

  @WorkerThread
  public synchronized void loadModel() {
    try {
      model = BertQuestionAnswerer.createFromFile(this.context, MODEL_PATH);
      Interpreter.Options opt = new Interpreter.Options();
      opt.setNumThreads(NUM_LITE_THREADS);
      Log.v(TAG, "TFLite model loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  @WorkerThread
  public synchronized List<org.tensorflow.lite.task.text.qa.QaAnswer> modelPredict(String query, String content) {
    List<org.tensorflow.lite.task.text.qa.QaAnswer> answers = model.answer(content, query);
    return answers;
  }
}
