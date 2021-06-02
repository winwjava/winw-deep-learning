package winw.ai.util.nlp;

import java.util.List;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

public class HankNlpTest {

	public static void main(String[] args) {
		Segment enablePlaceRecognize = HanLP.newSegment().enablePlaceRecognize(true);
		
		List<Term> seg = enablePlaceRecognize.seg("我家住在黄土高坡");
		System.out.println(seg);
		
	}
}
