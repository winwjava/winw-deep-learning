package winw.ai.util.nlp;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;

public class OpenNLPTest {

	public static void main(String[] args) throws IOException {
		
		// http://maven.tamingtext.com/opennlp-models/models-1.5/en-sent.bin
		String paragraph = "Hi. How are you? This is JD_Dog. He is my good friends.He is very kind.but he is no more handsome than me. ";
		InputStream is = new FileInputStream("E:\\NLP_Practics\\models\\en-sent.bin");
		SentenceModel model = new SentenceModel(is);
		SentenceDetectorME sdetector = new SentenceDetectorME(model);
		String sentences[] = sdetector.sentDetect(paragraph);
		for (String single : sentences) {
			System.out.println(single);
		}
		is.close();
	}

}
