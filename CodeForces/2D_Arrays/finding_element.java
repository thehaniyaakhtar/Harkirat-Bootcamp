import java.io.*;

public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        
        String[] first = br.readLine().split(" ");
        int n = Integer.parseInt(first[0]);
        int m = Integer.parseInt(first[1]);
        int x = Integer.parseInt(first[2]);
        
        boolean found = false;
        
        for(int i = 0; i < n; i++){
            String[] row = br.readLine().split(" ");
            
            for(int j = 0; j < m; j++){
                int num = Integer.parseInt(row[j]);
                
                if(num == x){
                    found = true;
                    break;
                }
            }
            if(found) break;
        }
        System.out.println(found);
    }
}
