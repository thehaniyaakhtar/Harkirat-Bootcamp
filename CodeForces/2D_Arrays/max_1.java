import java.util.*;
import java.io.*;

public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = new StringTokenizer(br.readLine());
        
        int n = Integer.parseInt(st.nextToken());
        int m = Integer.parseInt(st.nextToken());
        
        int maxCount = 0;
        int index = -1;
        
        for(int i = 0; i < n; i++){
            st = new StringTokenizer(br.readLine());
            
            int count = 0;
            
            for(int j = 0; j < m; j++){
                int val = Integer.parseInt(st.nextToken());
                if(val == 1){
                    count++;
                }
            }
            if(count > maxCount){
                maxCount = count;
                index = i;
            }
        }
        System.out.println(index);
    }
}
