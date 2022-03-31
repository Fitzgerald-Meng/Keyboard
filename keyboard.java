package com.example.magickeyboard.Activities;   //This part is mainly written by me 

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;

import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Color;
import android.icu.text.SimpleDateFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.TextPaint;
import android.text.method.LinkMovementMethod;
import android.text.style.BackgroundColorSpan;
import android.text.style.ClickableSpan;
import android.text.style.ForegroundColorSpan;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.ExecutorService;


import com.example.magickeyboard.R;
import com.example.magickeyboard.Utils.Decoder;
import com.example.magickeyboard.Utils.MyFileOperator;
import com.example.magickeyboard.Utils.Point;
import com.example.magickeyboard.View.DrawingView;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;

public class Typing extends AppCompatActivity {
    // logging tag used for debugging
    private static final String TAG = "TypingActivity";

    // services, operator, self-defined class objects
    private Decoder baselineDecoder;
    private static Socket socket;
    private MyFileOperator myFileOperator;
    private ExecutorService SocketService = Executors.newSingleThreadExecutor();

    // basic information about the test setup
    private Bundle FullInformation;
    protected int subjectID;
    protected String firstName;
    protected String lastName;
    protected int decoderNum;

    // socket communication
    private String raw_message;
    public String received_message = "";
    private ArrayList<Point> points;

    // UI components
    private TextView Subject;
    private TextView DecoderType;
    private TextView Progress;
    private TextView TargetPhrase;
    private TextView TranscribedPhrase;
    private DrawingView myDrawingCanvas;
    private ConstraintLayout myKeyboardContainer;
    private Button candidate1;
    private Button candidate2;
    private Button candidate3;

    // keyboard params
    private int keyboardWidth;
    private int keyboardHeight;
    private double g_x;
    private double g_y;
    private double Horizontal=1;
    private double Longitudinal=1.42;

    // variables for test progress and status
    private List<String> testPhrases;
    private int currentPhraseNum = 0;
    private ArrayList<String> candidateList;
    boolean isAutoCommitting = false;
    private int autoCommitNum = 0;
    String directoryName;
    String resultFileName;

    // jerk and speed calculation
    private int JERK_POINT_NUM_THRESHOLD = 150;
    ArrayList<Point> jerkPoints;
    double jerk_sum=0;
    double v_sum=0;
    int NumberOfJerk=0;
    int NumberOfV=0;
    int previousNum = - JERK_POINT_NUM_THRESHOLD;
    int currentNum=0;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getIntent().getExtras().getInt("Decoder") == 1) {
            setContentView(R.layout.activity_typing_baseline);
        }
        else {
            setContentView(R.layout.activity_typing);
        }

        initialize();
    }

    private char getCharAtIndex(String target, int index){
        if (!Character.isLetter(target.charAt(index))){
            return '#';
        }
        return target.charAt(index);
    }

    private void updateDisplay(TextView targetView, String label, String value){    //This main function is written by me and used to draw the keyboard and record the data
        String tmp = label + ": " + value;
        Spannable display = new SpannableString(tmp);
        display.setSpan(new ForegroundColorSpan(getResources().getColor(R.color.colorPrimary)), 0, label.length() + 1, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
        display.setSpan(new ForegroundColorSpan(Color.BLACK), label.length() + 1, tmp.length() - 1, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
        targetView.setText(display);
    }

    // initializing the UI, preparation of File I/O;
    private void initialize(){
        // initialize the UI;
        labelsInitialize();

        // set the target directory name and file name
        directoryName = "Subject_" + subjectID + "_" +firstName + "_" + lastName;
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault());
        if (decoderNum == 1) {
            resultFileName = "SubejectID_" + subjectID + "_Baseline_"  + sdf.format(new Date()) + ".txt";
        }
        else {
            resultFileName = "SubejectID_" + subjectID + "_NewDecoder_" + sdf.format(new Date()) + ".txt";
        }

        // create an object of the File I/O wrapper class
        myFileOperator = new MyFileOperator(this);
        myFileOperator.createFile(directoryName, resultFileName);

        // load the test phrases and display the first target phrase on the interface
        // loadTestCaseFile();
        loadTestCaseFile(decoderNum);
        TargetPhrase.setText(testPhrases.get(currentPhraseNum));
        TranscribedPhrase.setText("");

        myDrawingCanvas = (DrawingView) findViewById(R.id.drawing);
        myKeyboardContainer = (ConstraintLayout) findViewById(R.id.keyboard);

        ViewTreeObserver vto = myKeyboardContainer.getViewTreeObserver();
        vto.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                myKeyboardContainer.getViewTreeObserver().removeGlobalOnLayoutListener(this);
                int height = myKeyboardContainer.getMeasuredHeight();
                int width = myKeyboardContainer.getMeasuredWidth();

                ConstraintLayout.LayoutParams current_params = (ConstraintLayout.LayoutParams) myDrawingCanvas.getLayoutParams();

                current_params.height = (height - 1) * 3 / 4; // a little bit smaller
                myDrawingCanvas.setLayoutParams(current_params);

                keyboardHeight = height;
                keyboardWidth = width;

                g_x = width / 2.0;
                g_y = (height * 3.0) / 8.0;

                if (decoderNum == 1) {
                    try {
                        baselineDecoder = new Decoder(Typing.this, keyboardWidth, keyboardHeight);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        });

        if (decoderNum == 2) {
            SocketService.execute(socketService);
        }
        else {
            myDrawingCanvas.setOnTouchListener(new View.OnTouchListener() {
                @Override
                public boolean onTouch(View v, MotionEvent event) {
                    points = myDrawingCanvas.getPoints();

                    currentNum = points.size();

                    double s = -100;
                    double[] min_v = {-100};

                    if(points.size()>7) {
                        jerkPoints = new ArrayList<Point>();
                        for (int i = 7; i >= 1; i--) {
                            jerkPoints.add(points.get(points.size() - i));
                        }

                        s=minimum_jerk();
                        jerk_sum+=s;
                        min_v = minimum_v();
                        v_sum+=min_v[3];
                    }

                    if(event.getAction() == MotionEvent.ACTION_DOWN) {
                        if (isAutoCommitting) clearAutoCommitInfo();
                    }

                    if(event.getAction() == MotionEvent.ACTION_UP) {
//                            || ((((points.size() > 7) && (s < (jerk_sum / (3 * NumberOfJerk)) && s > 0)
//                            && ((currentNum-previousNum) > 150)))
//                            &&(min_v[0]<v_sum/NumberOfV && min_v[0] > 0))) {

                        if(event.getAction() == MotionEvent.ACTION_UP) {
                            clearJerkVariables();
                            Log.i(TAG, "Reset previous Num");
                        }
                        else {
                            previousNum = currentNum;
                        }

                        points = myDrawingCanvas.getPoints();

                        if (event.getAction() == MotionEvent.ACTION_UP) {
                            if (isSameLetter() == '!') {
                                new Thread(){
                                    public void run(){
                                        received_message = String.join("@@@", baselineDecoder.decode(points));
                                        received_message += "$";
                                        mHandler.sendMessage(mHandler.obtainMessage());
                                    }
                                }.start();
                            }
                            else {
                                received_message = isSameLetter() + "$";
                                mHandler.sendMessage(mHandler.obtainMessage());
                            }

                        }
                        else {
                            new Thread(){
                                public void run(){
                                    received_message = String.join("@@@", baselineDecoder.decode(points));
                                    mHandler.sendMessage(mHandler.obtainMessage());
                                }
                            }.start();
                        }
                    }
                    return false;
                }
            });
        }

        updatePhraseNumDisplay();
    }

    public Handler mHandler = new Handler() {
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            Log.i("mHandler", "Being called for: " + received_message);
            processReceivedMessage();

            // check if initial entry or subsequent edit

        }
    };

    private void processReceivedMessage(){
        boolean isLastMessage = false;

        if (received_message.length() == 0) return;

        if (received_message.endsWith("$")) {
            if (received_message.length() == 1) return;
            received_message = received_message.substring(0, received_message.length() - 1);
            isLastMessage = true;
        }

        System.out.println(received_message.split("@@@"));
        candidateList = new ArrayList<String>(Arrays.asList(received_message.split("@@@")));

        if (candidateList.size() > 0)   candidate1.setText(candidateList.get(0));
        if (candidateList.size() > 1)   candidate2.setText(candidateList.get(1));
        if (candidateList.size() > 2)   candidate3.setText(candidateList.get(2));

        Bundle editMode = getIntent().getExtras();
        if (!isEditing(editMode) && candidateList.size() > 0 && isLastMessage)  {
            // above we use NumberOfJerk to judge whether the message is the final one (i.e., current gesture is finished)
            // since we only auto-commit the final one
            appendTranscribedPhrases(editMode, candidateList.get(0));
            isAutoCommitting = true;
            autoCommitNum = Arrays.asList(candidateList.get(0).split(" ")).size();
        }
    }

    Runnable socketService = new Runnable(){                //This function is written by me and it is used to transfer data between the android app and the server
        private Socket socket = null;
        private BufferedReader in = null;
        private PrintWriter out = null;
        private static final String HOST = "192.168.0.31";
        private static final int PORT = 8000;
        private String msg;
        private ExecutorService SendService = Executors.newSingleThreadExecutor(); //创建单线程池，用于发送数据

        @Override
        public void run(){
            try {
                socket = new Socket(HOST, PORT);
                in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(socket.getOutputStream())), true);

                while (true) {
                    myDrawingCanvas.setOnTouchListener(new View.OnTouchListener() {
                        @Override
                        public boolean onTouch(View v, MotionEvent event) {
                            points = myDrawingCanvas.getPoints();

                            currentNum = points.size();

                            double s = -100;
                            double[] min_v = {-100};

                            if(points.size()>7) {
                                jerkPoints = new ArrayList<Point>();
                                for (int i = 7; i >= 1; i--) {
                                    jerkPoints.add(points.get(points.size() - i));
                                }

                                s=minimum_jerk();
                                jerk_sum+=s;
                                min_v = minimum_v();
                                v_sum+=min_v[3];
                            }

                            if(event.getAction() == MotionEvent.ACTION_DOWN) {
                                if (isAutoCommitting) clearAutoCommitInfo();
                            }

                            if(event.getAction() == MotionEvent.ACTION_UP
                                    || ((((points.size() > 7) && (s < (jerk_sum / (3 * NumberOfJerk)) && s > 0)
                                    && ((currentNum-previousNum) > 150)))
                                    &&(min_v[0]<v_sum/NumberOfV && min_v[0] > 0))) {

                                if(event.getAction() == MotionEvent.ACTION_UP) {
                                    clearJerkVariables();
                                    Log.i(TAG, "Reset previous Num");
                                    raw_message = "$";
                                }
                                else {
                                    raw_message = "";
                                    previousNum = currentNum;
                                }

                                points = myDrawingCanvas.getPoints();
                                raw_message += processPoints();
                                raw_message += "###";

//                                if (decoderNum == 1) {
//                                    Log.i("Baseline Decoding: ", baselineDecoder.decode(points));
//                                }

                                Log.i(TAG, "Ready to send: " + raw_message);

                                if (socket.isConnected() && !socket.isOutputShutdown()) {
                                    if(out != null) {
                                        msg = raw_message;
                                        Long startTime = System.currentTimeMillis();
                                        String st = startTime.toString();
                                        Log.i(TAG, "Timestamp before sending to server: " + st);
                                        SendService.execute(sendService);
                                    }
                                }
                            }
                            return false;
                        }
                    });

                    if (!socket.isClosed() && socket.isConnected() && !socket.isInputShutdown()) {
                        try {
                            if ((received_message = in.readLine()) != null) {
                                Long endTime = System.currentTimeMillis();
                                String et = endTime.toString();
                                Log.i(TAG, "Timestamp after receiving from server: " + et);
                                mHandler.sendMessage(mHandler.obtainMessage());
                            }
                        }
                        catch (IOException e) {
                            Log.d("send",Log.getStackTraceString(e));
                        }
                        catch(NullPointerException e)
                        {
                            Log.d("send",Log.getStackTraceString(e));
                        }
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        Runnable sendService = new Runnable(){
            @Override
            public void run() {
                out.println(msg);   //通过输出流发送数据
            }
        };
    };

    public void exit(View view){
        finish();
    }

    // HW START
    private Boolean isEditMode(Bundle editMode){
        // return true if system in edit mode
        return editMode.getBoolean("editable");
    }

    private Boolean isEditing(Bundle editMode){
        if (!editMode.getBoolean("editable")) {
            return false;
        }

        ArrayList<String> listOfStr = editMode.getStringArrayList("arrOfStr");
        for (int i = 0; i <= listOfStr.size(); i++) {
            if (editMode.getInt(String.valueOf(i)) == 1){
                return true;
            }
        }

        return false;
    }

    private void appendTranscribedPhrases(Bundle editMode, String incoming_message){
        // update new string from server
        editMode.putBoolean("editable", true); // switch to edit mode

        String[] arrOfStr = incoming_message.split(" ");
        ArrayList<String> newListOfStr = new ArrayList<String>(Arrays.asList(arrOfStr));
        ArrayList<String> listOfStr = editMode.getStringArrayList("arrOfStr");
        if (listOfStr == null) listOfStr = new ArrayList<String>();
        listOfStr.addAll(newListOfStr);
        editMode.putStringArrayList("arrOfStr", listOfStr);

        Intent intent = getIntent();
        intent.putExtras(editMode);

        String transcribedPhrase = String.join(" ", listOfStr);

        resetSpan(listOfStr, transcribedPhrase);
    }

    private void replaceTranscribedPhrase(Bundle editMode, String editedWord){
        if (!isEditMode(editMode)) return;

        // replace existing phrase
        Log.i(TAG, "Editing...");
        List<Integer> targetPosList = new ArrayList<>();
        ArrayList<String> listOfStr = editMode.getStringArrayList("arrOfStr");
        for (int i = 0; i < listOfStr.size(); i++) {
            int isTargetPos = editMode.getInt(String.valueOf(i));
            if (isTargetPos == 1){
                Log.i(TAG, "Will edit: " + String.valueOf(i) + " " + listOfStr.get(i));
                targetPosList.add(i);
                editMode.putInt(String.valueOf(i), 0);
            }
        }
        if (!targetPosList.isEmpty()){ // check if an edit has been made
            listOfStr.set(targetPosList.get(0), editedWord);
            targetPosList.remove(0);
            Log.i(TAG, "Indices to delete..." + targetPosList.toString());

            Log.i(TAG, "Word list before: " + listOfStr.toString());
            removeIndices(listOfStr, targetPosList);
            Log.i(TAG, "Word list after: " + listOfStr.toString());

            String editedSentence = String.join(" ", listOfStr);
            String[] arrOfStr = editedSentence.split(" ");
            listOfStr = new ArrayList<String>(Arrays.asList(arrOfStr));
            resetSpan(listOfStr, editedSentence);
            // if break up word into two
            Intent intent = getIntent();
            editMode.putStringArrayList("arrOfStr", listOfStr);
            intent.putExtras(editMode);
        }
    }

    private void removeIndices(List<String> other, List<Integer> indices)
    {
        indices.stream()
                .sorted(Comparator.reverseOrder())
                .forEach(i->other.remove(i.intValue()));
    }

    private void resetSpan(ArrayList<String> arrOfStr, String text){
        SpannableStringBuilder ss = new SpannableStringBuilder(text);
        int startIndex = 0, endIndex = -1;
        for (int i = 0; i < arrOfStr.size(); i++) {
            if (endIndex != -1){
                startIndex = text.indexOf(arrOfStr.get(i), endIndex);
            }
            else{ startIndex = text.indexOf(arrOfStr.get(i));
            }
            endIndex = startIndex + arrOfStr.get(i).length();
            ss.setSpan(new CustomClickableSpan(), startIndex, endIndex, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        TranscribedPhrase.setText(ss, TextView.BufferType.EDITABLE);
        TranscribedPhrase.setMovementMethod(LinkMovementMethod.getInstance());
    }

    private class CustomClickableSpan extends ClickableSpan {
        private boolean highlightWord = false;

        @Override
        public void onClick(@NonNull View widget) {
            // TODO add check if widget instanceof TextView
            if (isAutoCommitting) clearAutoCommitInfo();

            TextView tv = (TextView) widget;
            tv.setHighlightColor(Color.TRANSPARENT); // remove default highlight
            SpannableStringBuilder s = (SpannableStringBuilder) tv.getText();
            int start = s.getSpanStart(this);
            int end = s.getSpanEnd(this);
            Log.d(TAG, "onClick [" + s.subSequence(start, end) + "]");

            Bundle editMode = getIntent().getExtras();
            ArrayList<String> listOfStr = editMode.getStringArrayList("arrOfStr");
            String target = String.valueOf(s.subSequence(start, end));
            int targetPos = listOfStr.indexOf(target);
            // check for repeated word
//            int[] matchingIndices = IntStream.range(0, listOfStr.size())
//                    .filter(i -> target.equals(listOfStr.get(i)))
//                    .toArray();
//            Log.d(TAG, "Possible clicked positions: " + Arrays.toString(matchingIndices));
            ArrayList<Integer> matchingIndices = findEqualOrSuffix(listOfStr, target);
            Log.d(TAG, "Possible clicked positions: " + matchingIndices.toString());

            int matchingIndex = tv.getText().toString().indexOf(target);
            int selectedIndex = 0;
            int matchLength = target.length();
            while (matchingIndex >= 0) {  // indexOf returns -1 if no match found
                if (start == matchingIndex){
                    Log.d(TAG, "Match at location " + String.valueOf(matchingIndex));
                    targetPos = matchingIndices.get(selectedIndex);
                }
                else{ selectedIndex += 1;
                }
                matchingIndex = tv.getText().toString().indexOf(target, matchingIndex + matchLength);
            }

            Intent intent = getIntent();
            // check if valid click
            boolean newEdit = true;
            for (int i = 0; i <= listOfStr.size(); i++) {
                if (editMode.getInt(String.valueOf(i)) == 1){
                    newEdit = false;
                    break;
                }
            }
            if (newEdit){
                editMode.putInt(String.valueOf(targetPos), 1);
                Log.d(TAG, "Valid position: " + String.valueOf(targetPos));
                highlightWord = true;
            }else{
                // cancel if clicked twice
                if (editMode.getInt(String.valueOf(targetPos)) == 1){
                    editMode.putInt(String.valueOf(targetPos), 0);
                    Log.d(TAG, "Unselected position: " + String.valueOf(targetPos));
                    highlightWord = false;
                    // remove highlight
                    s.setSpan(new BackgroundColorSpan(Color.TRANSPARENT), start, end, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
                }
                else if ((targetPos-1 >= 0 && editMode.getInt(String.valueOf(targetPos-1)) == 1)
                        || (targetPos+1 < listOfStr.size() && editMode.getInt(String.valueOf(targetPos+1)) == 1)){
                    // if next to valid selection
                    editMode.putInt(String.valueOf(targetPos), 1);
                    Log.d(TAG, "Valid position: " + String.valueOf(targetPos));
                    highlightWord = true;
                }
            }

            if (highlightWord){ // set highlight
                s.setSpan(new BackgroundColorSpan(Color.parseColor("#ff8fb6")), start, end, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
            }
            intent.putExtras(editMode);
        }

        @Override
        public void updateDrawState(TextPaint ds) {
            ds.setARGB(255, 0, 0, 0);
            // remove default underline associated with spans
            ds.setUnderlineText(false);
        }
    }

    private ArrayList<Integer> findEqualOrSuffix(ArrayList<String> listOfStr, String target) {
        ArrayList<Integer> result = new ArrayList<Integer>();

        for (int i = 0; i < listOfStr.size(); i++) {
            if (target.equals(listOfStr.get(i)) || listOfStr.get(i).contains(target)) {
                result.add(i);
            }
        }

        return result;
    }
    // HW END

    private String processPoints(){                         //This function is written by me and it is used to process the given points.
        String result = "";
        float x, y, original_x, original_y;
        for (Point point : points) {
            original_x = point.getX();
            original_y = point.getY();

            if (original_y < 0) original_y = 0;
            if (original_y > keyboardHeight) original_y = keyboardHeight;
            x = (float) ((original_x - g_x) / (keyboardWidth / 10));
            y = (float) ((g_y - original_y) / (keyboardWidth / 10));
            result += getNearestLetter(x, y);
        }

        String compressed_result = "";
        int i = 0;
        int count = 0;
        char last_letter = '!';

        while(i < result.length()) {
            if (result.charAt(i) != last_letter) {
                if (count != 0) {
                    compressed_result += Integer.toString(count);
                }
                count = 1;
                compressed_result += result.charAt(i);
                last_letter = result.charAt(i);
            }
            else {
                count++;
            }
            i++;
        }

        compressed_result += Integer.toString(count);

        return compressed_result;
    }

    private char isSameLetter(){
        String result = "";
        float x, y, original_x, original_y;
        for (Point point : points) {
            original_x = point.getX();
            original_y = point.getY();

            if (original_y < 0) original_y = 0;
            if (original_y > keyboardHeight) original_y = keyboardHeight;
            x = (float) ((original_x - g_x) / (keyboardWidth / 10));
            y = (float) ((g_y - original_y) / (keyboardWidth / 10));
            result += getNearestLetter(x, y);
        }

        int i = 0;

        char first_letter = result.charAt(0);

        while(i < result.length()) {
            if (result.charAt(i) != first_letter) {
                return '!';
            }
            i++;
        }

        return first_letter;
    }

    private void labelsInitialize(){                //This sentence is partly written by me and it is used to initialize labels
        // restore all basic information (about the test and subject) collected in the previous activities
        Intent intent = getIntent();
        FullInformation = getIntent().getExtras();
        firstName = FullInformation.getString("FirstName");
        lastName = FullInformation.getString("LastName");
        subjectID = FullInformation.getInt("SubjectID");
        decoderNum = FullInformation.getInt("Decoder");

        resetEditInfo();

        Progress = findViewById(R.id.ProgressField);
        DecoderType = findViewById(R.id.DecoderField);
        Subject = findViewById(R.id.SubjectField);
        TargetPhrase = findViewById(R.id.target_phrase);
        TranscribedPhrase = findViewById(R.id.transcribed_phrase);
        candidate1 = findViewById(R.id.candidate_1);
        candidate2 = findViewById(R.id.candidate_2);
        candidate3 = findViewById(R.id.candidate_3);
        candidate1.setText("");
        candidate2.setText("");
        candidate3.setText("");

        updateDisplay(DecoderType, "Decoder", (decoderNum == 1) ? "Baseline" : "New");
        updateDisplay(Subject, "Subject", Integer.toString(subjectID));
    }

    //输出句子的进度
    private void updatePhraseNumDisplay() {
        updateDisplay(Progress, "Progress", currentPhraseNum + "/" + (testPhrases.size() - currentPhraseNum));
    }

    private void loadTestCaseFile(int file_index) {
        testPhrases = new ArrayList<>();
        String testFileName = "group" + file_index + ".txt";
        testPhrases = myFileOperator.readFileIntoStringList(testFileName);
    }

    public void candidateHandler(View view) {
        int selectID = 0;
        switch (view.getId()) {
            case  R.id.candidate_1: {
                selectID = 0;
                if (candidate1.getText() == "") return;
                break;
            }
            case R.id.candidate_2: {
                selectID = 1;
                if (candidate2.getText() == "") return;
                break;
            }
            case R.id.candidate_3: {
                selectID = 2;
                if (candidate3.getText() == "") return;
                break;
            }
        }

        Bundle editMode = getIntent().getExtras();
        if (!isEditing(editMode)) {
            if (selectID == 1 || selectID == 2) {
                ArrayList<String> listOfStr = editMode.getStringArrayList("arrOfStr");
                List<Integer> deletePosition = new ArrayList<Integer>();
                for (int i = listOfStr.size() - 1; i >= listOfStr.size() - autoCommitNum; i--) {
                    deletePosition.add(i);
                }

                removeIndices(listOfStr, deletePosition);
                resetSpan(listOfStr, String.join(" ", listOfStr));

                appendTranscribedPhrases(editMode, candidateList.get(selectID));
            }

            if (isAutoCommitting) clearAutoCommitInfo();
        }
        else {
            replaceTranscribedPhrase(editMode, candidateList.get(selectID));
        }

        candidate1.setText("");
        candidate2.setText("");
        candidate3.setText("");

        candidateList.clear();
    }

    public void deleteHandler(View view) {
        Bundle editMode = getIntent().getExtras();
        ArrayList<String> listOfStr = editMode.getStringArrayList("arrOfStr");
        if (listOfStr == null || listOfStr.size() == 0 || TranscribedPhrase.getText() == "") return;

        if (isAutoCommitting) clearAutoCommitInfo();

        if (!isEditing(editMode)) {
            List<Integer> deletePosition = new ArrayList<Integer>();
            deletePosition.add(listOfStr.size() - 1);

            removeIndices(listOfStr, deletePosition);

            if (listOfStr.size() != 0) {
                resetSpan(listOfStr, String.join(" ", listOfStr));
            }
            else {
                TranscribedPhrase.setText("");
                editMode.putBoolean("editable", false);
            }
        }
        else {
            List<Integer> deletePosition = new ArrayList<Integer>();

            for (int i = 0; i < listOfStr.size(); i++) {
                int isTargetPos = editMode.getInt(String.valueOf(i));
                if (isTargetPos == 1){
                    deletePosition.add(i);
                    editMode.putInt(String.valueOf(i), 0);
                }
            }

            removeIndices(listOfStr, deletePosition);

            if (listOfStr.size() != 0) {
                resetSpan(listOfStr, String.join(" ", listOfStr));
            }
            else {
                TranscribedPhrase.setText("");
                editMode.putBoolean("editable", false);
            }
        }

        getIntent().putExtras(editMode);
    }

    public void skip10Handler(View view){                   //This function is written by me and it is used to skip next 10 sentences
        if ((currentPhraseNum + 10) > testPhrases.size() - 1) return;
        currentPhraseNum += 10;

        // update the phrase and number display on the UI;
        updatePhraseNumDisplay();
        TargetPhrase.setText(testPhrases.get(currentPhraseNum));
        TranscribedPhrase.setText("");

        // clear edit info;
        resetEditInfo();

        //clear auto-commiting info
        if (isAutoCommitting) clearAutoCommitInfo();
    }

    public void skipHandler(View view){                     //This function is written by me and it is used to skip next sentences
        if ((currentPhraseNum + 1) > testPhrases.size() - 1) return;
        currentPhraseNum += 1;

        // update the phrase and number display on the UI;
        updatePhraseNumDisplay();
        TargetPhrase.setText(testPhrases.get(currentPhraseNum));
        TranscribedPhrase.setText("");

        // clear edit info;
        resetEditInfo();

        //clear auto-commiting info
        if (isAutoCommitting) clearAutoCommitInfo();
    }

    public void nextHandler(View view){                     //This function is written by me and it is used to go to next sentences
        // update target phrase display and progress
        if ((currentPhraseNum + 1) > testPhrases.size() - 1) return;
        currentPhraseNum += 1;
        TargetPhrase.setText(testPhrases.get(currentPhraseNum));
        updatePhraseNumDisplay();

        // clear transcribed phrase and editing info
        TranscribedPhrase.setText("");
        resetEditInfo();

        //clear auto-commiting info
        if (isAutoCommitting) clearAutoCommitInfo();
    }

    private String getNearestLetter(float x, float y)   //This function is written by me and it is used to get the nearest letter given a coordinator.
    {
        if(y<-0.5*Longitudinal) {
            if ((-0.5 * Horizontal <x||-0.5*Horizontal==x)&&x <0.5*Horizontal)
            {
                return "v";
            }
            else if((0.5 * Horizontal <x||0.5*Horizontal==x)&&x <1.5*Horizontal)
            {
                return "b";
            }
            else if((1.5 * Horizontal <x||1.5*Horizontal==x)&&x <2.5*Horizontal)
            {
                return "n";
            }
            else if(2.5 * Horizontal <x||2.5*Horizontal==x)
            {
                return "m";
            }
            else if((-1.5 * Horizontal <x||-1.5*Horizontal==x)&&x <-0.5*Horizontal)
            {
                return "c";
            }
            else if((-2.5 * Horizontal <x||-2.5*Horizontal==x)&&x <-1.5*Horizontal)
            {
                return "x";
            }
            else if(x <-2.5*Horizontal)
            {
                return "z";
            }
        }
        else if(y>Longitudinal*0.5){
            if((x>0||x==0)&&x<1*Horizontal)
            {
                return "y";
            }
            else if((1 * Horizontal <x||1*Horizontal==x)&&x <2*Horizontal)
            {
                return "u";
            }
            else if((2 * Horizontal <x||-2*Horizontal==x)&&x <3*Horizontal)
            {
                return "i";
            }
            else if((3 * Horizontal <x||3*Horizontal==x)&&x <4*Horizontal)
            {
                return "o";
            }
            else if(4 * Horizontal <x||4*Horizontal==x)
            {
                return "p";
            }
            else if((-Horizontal<x||-Horizontal==x)&&x<0)
            {
                return "t";
            }
            else if((-2 * Horizontal <x||-2*Horizontal==x)&&x <(-1)*Horizontal)
            {
                return "r";
            }
            else if((-3 * Horizontal <x||-3*Horizontal==x)&&x <-2*Horizontal)
            {
                return "e";
            }
            else if((-4 * Horizontal <x||-4*Horizontal==x)&&x <-3*Horizontal)
            {
                return "w";
            }
            else if(x <-4*Horizontal)
            {
                return "q";
            }
        }
        else
        {
            if((-0.5 * Horizontal <x||-0.5*Horizontal==x)&&x <0.5*Horizontal)
            {
                return "g";
            }
            else if((0.5 * Horizontal <x||0.5*Horizontal==x)&&x <1.5*Horizontal)
            {
                return "h";
            }
            else if((1.5 * Horizontal <x||1.5*Horizontal==x)&&x <2.5*Horizontal)
            {
                return "j";
            }
            else if((2.5 * Horizontal <x||2.5*Horizontal==x)&&x <3.5*Horizontal)
            {
                return "k";
            }
            else if(x>3.5*Horizontal||x==3.5*Horizontal)
            {
                return "l";
            }
            else if((-1.5 * Horizontal <x||-1.5*Horizontal==x)&&x <-0.5*Horizontal)
            {
                return "f";
            }
            else if((-2.5 * Horizontal <x||-2.5*Horizontal==x)&&x <-1.5*Horizontal)
            {
                return "d";
            }
            else if((-3.5 * Horizontal <x||-3.5*Horizontal==x)&&x <-2.5*Horizontal)
            {
                return "s";
            }
            else if(x<-3.5*Horizontal)
            {
                return "a";
            }
        }
        return "";
    }

    public void resetEditInfo(){
        Bundle editMode = new Bundle();
        editMode.putBoolean("editable", false);
        editMode.putString("FirstName", firstName);
        editMode.putString("LastName", lastName);
        editMode.putInt("SubjectID", subjectID);
        editMode.putInt("Decoder", decoderNum);
        getIntent().replaceExtras(editMode);
    }

    private double minimum_jerk()   //This function is written by me and it is used to calculate the minimum jerk
    {
        ArrayList<Double> v = new ArrayList<Double>();
        for (int i = 0 ; i < 5; i++) {
            v.add((Math.sqrt(Math.pow(jerkPoints.get(i).getX()-jerkPoints.get(i + 2).getX(),2) +
                    Math.pow(jerkPoints.get(i).getY()-jerkPoints.get(i + 2).getY(),2)))/10);
        }

        double a3=1000*(v.get(2) - v.get(0))/10;
        double a5=1000*(v.get(4) - v.get(2))/10;
        double jerk=(a5-a3)/10;

        NumberOfJerk++;
        return jerk;
    }

    private double[] minimum_v()    //This function is written by me and it is used to calculate the velocity
    {
        double[] min_v=new double[4];

        ArrayList<Double> v = new ArrayList<Double>();
        for (int i = 0 ; i < 5; i++) {
            v.add((Math.sqrt(Math.pow(jerkPoints.get(i).getX()-jerkPoints.get(i + 1).getX(),2) +
                    Math.pow(jerkPoints.get(i).getY()-jerkPoints.get(i + 1).getY(),2)))/10);
        }

        min_v[0] = v.get(0);
        min_v[1] = v.get(1);
        min_v[2] = v.get(2);
        min_v[3] = (v.get(0) + v.get(1) + v.get(2))/3;
        NumberOfV++;
        return min_v;
    }

    private void clearJerkVariables(){      //This function is written by me and it is used tho clear all the variables
        jerk_sum=0;
        v_sum=0;
        NumberOfJerk=0;
        NumberOfV=0;
        previousNum = - JERK_POINT_NUM_THRESHOLD;
        currentNum=0;
    }

    private void clearAutoCommitInfo() {
        autoCommitNum = 0;
        isAutoCommitting = false;
        candidate1.setText(" ");
        candidate2.setText(" ");
        candidate3.setText(" ");
    }
}
