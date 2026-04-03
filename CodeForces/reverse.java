import java.util.*;
import java.io.*;

public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String s = br.readLine();
        
        StringBuilder sb = new StringBuilder();
        
        for(int i = s.length()-1; i >=0; i--){
            sb.append(s.charAt(i));
        }
        System.out.print(sb.toString());
    }
}
