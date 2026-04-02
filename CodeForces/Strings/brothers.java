import java.io.*;
import java.util.*;

public class Main{
    public static void main(String[] args) throws Exception{
        
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st1 = new StringTokenizer(br.readLine());
        StringTokenizer st2 = new StringTokenizer(br.readLine());
        
        String first1 = st1.nextToken();
        String last1 = st1.nextToken();
        
        String first2 = st2.nextToken();
        String last2 = st2.nextToken();
        
        if(last1.equals(last2)){
            System.out.println("YES");
        }
        else{
            System.out.println("NO");
        }
        
    }
}
