import java.util.*;
import java.io.*;

public class Main{
    public static void main(String[] args) throws Exception{
        
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        
        String s = br.readLine();
        String[] parts = br.readLine().split(" ");
        
        char c1 = parts[0].charAt(0);
        
        String result = s.replace(String.valueOf(c1), "");
        
        System.out.println(result);
    }
}
