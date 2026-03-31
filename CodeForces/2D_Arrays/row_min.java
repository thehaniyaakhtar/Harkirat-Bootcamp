import java.io.*;
import java.util.*;

public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = new StringTokenizer(br.readLine());
        
        int n = Integer.parseInt(st.nextToken());
        int m = Integer.parseInt(st.nextToken());
        
        StringBuilder result = new StringBuilder();
        
        for(int i = 0; i < n; i++){
            st = new StringTokenizer(br.readLine());
            
            int min = Integer.MAX_VALUE;
            
            for(int j = 0; j < m; j++){
                int val = Integer.parseInt(st.nextToken());
                min = Math.min(min, val);
            }
            result.append(min).append(" ");
        }
        System.out.println(result.toString().trim());
    }
}
